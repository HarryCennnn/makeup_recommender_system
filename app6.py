import pymysql
pymysql.install_as_MySQLdb()
from flask import Flask, render_template, request, redirect, url_for, session
from sqlalchemy import create_engine, text
import random
import pandas as pd
import numpy as np
from functools import wraps
import bcrypt
import json
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ---- 连接 MySQL 数据库 ----
conn_string = 'mysql+pymysql://{user}:{password}@{host}:{port}/{db}?charset=utf8'.format(
    user="dbmh",
    password="jelIyvfzqwNBWJXpC9bJTQ==",
    host='jsedocc7.scrc.nyu.edu',
    port=3306,
    db='dbmh'
)
engine = create_engine(conn_string)

# ---- Login required decorator ----
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'loggedin' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ---- Register route ----
@app.route('/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        query = text("SELECT * FROM user_accounts WHERE username = :username")
        with engine.connect() as conn:
            result = conn.execute(query, {'username': username}).fetchone()

        if result:
            msg = 'Username already exists!'
        else:
            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            insert_query = text("INSERT INTO user_accounts (username, password) VALUES (:username, :password)")
            with engine.begin() as conn:
                conn.execute(insert_query, {'username': username, 'password': hashed_pw})
            return redirect(url_for('login'))

    return render_template('register.html', msg=msg)

# ---- Login route ----
@app.route('/login', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        query = text("SELECT * FROM user_accounts WHERE username = :username")
        with engine.connect() as conn:
            account = conn.execute(query, {'username': username}).fetchone()

        if account and bcrypt.checkpw(password.encode('utf-8'), account[2].encode('utf-8')):
            session['loggedin'] = True
            session['id'] = account[0]
            session['username'] = account[1]
            return redirect(url_for('index'))
        else:
            msg = 'Incorrect username or password!'

    return render_template('login.html', msg=msg)

# ---- Logout route ----
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ---- Protected index route ----
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# ---- 全局变量 ----
filtered_products = []
reorder_products = []

# ---- 筛选产品类别 ----
@app.route('/select_category', methods=['POST'])
@login_required
def select_category():
    global filtered_products
    category = request.form['product_category']
    on_sale = request.form['is_on_sale']
    free_shipping = request.form['is_free_shipping']
    fast_delivery = request.form['is_3day_delivery']

    # 初始查询语句 & 参数
    query_str = "SELECT * FROM Makeup_Table WHERE product_category = :category"
    params = {'category': category}

    # 只有当用户明确选择了 "Yes" 时，才加入筛选条件
    if on_sale == "1":
        query_str += " AND is_on_sale = :on_sale"
        params['on_sale'] = 1

    if free_shipping == "1":
        query_str += " AND is_free_shipping = :free_shipping"
        params['free_shipping'] = 1

    if fast_delivery == "1":
        query_str += " AND is_3day_delivery = :fast_delivery"
        params['fast_delivery'] = 1

    # 执行查询
    query = text(query_str)
    with engine.connect() as conn:
        result = conn.execute(query, params)
        products = result.fetchall()

    # 构建产品列表
    filtered_products = [
        {
            'product_name': p[0], 'brand': p[1], 'current_price': p[2], 'unit_price': p[3],
            'star_rating': p[4], 'review_counts': p[5], 'is_on_sale': p[6], 'is_free_shipping': p[7],
            'is_3day_delivery': p[8], 'product_category': p[9], 'product_link': p[10], 'platform': p[11]
        } for p in products
    ]

    return redirect(url_for('order'))


# 可视化路由
@app.route('/visualization', methods=['POST'])
@login_required
def visualization():
    category = request.form.get('product_category')
    on_sale = request.form.get('is_on_sale')
    free_shipping = request.form.get('is_free_shipping')
    fast_delivery = request.form.get('is_3day_delivery')

    # 构建查询语句
    query_str = "SELECT current_price, review_counts, unit_price FROM Makeup_Table WHERE product_category = :category"
    params = {'category': category}

    if on_sale == "1":
        query_str += " AND is_on_sale = :on_sale"
        params['on_sale'] = 1
    if free_shipping == "1":
        query_str += " AND is_free_shipping = :free_shipping"
        params['free_shipping'] = 1
    if fast_delivery == "1":
        query_str += " AND is_3day_delivery = :fast_delivery"
        params['fast_delivery'] = 1

    # 查询数据
    query = text(query_str)
    with engine.connect() as conn:
        result = conn.execute(query, params)
        rows = result.fetchall()

    # 提取字段
    price_data = [float(row[0]) for row in rows if row[0] is not None]
    review_data = [int(row[1]) for row in rows if row[1] is not None]
    unit_price_data = [float(row[2]) for row in rows if row[2] is not None]

    return render_template(
        'visualization.html',
        category=category,
        on_sale=on_sale,
        free_shipping=free_shipping,
        fast_delivery=fast_delivery,
        price_data=json.dumps(price_data),
        review_data=json.dumps(review_data),
        unit_price_data=json.dumps(unit_price_data)
    )

# ---- 打分页面 ----
@app.route('/order', methods=['GET'])
@login_required
def order():
    return render_template('order.html')

# ---- 用户提交排序，计算 initial_weights 并生成 reorder 产品 ----
@app.route('/submit_ranking', methods=['POST'])
@login_required
def submit_ranking():
    global reorder_products, initial_weights

    session['attribute_ranking'] = {
        attr: int(request.form.get(attr)) for attr in ['star_rating', 'review_counts', 'current_price', 'unit_price']
    }

    ranking_attributes = ['star_rating', 'review_counts', 'current_price', 'unit_price']
    lower_is_better = ['current_price', 'unit_price']
    user_order = session['attribute_ranking']

    # 用户权重
    max_rank = len(user_order)
    user_weights = {attr: (max_rank - rank + 1) for attr, rank in user_order.items()}
    total = sum(user_weights.values())
    user_weights = {k: v / total for k, v in user_weights.items()}

    # 系统权重
    query = text("""
        SELECT weight_star_rating, weight_review_counts, weight_current_price, weight_unit_price
        FROM user_accounts WHERE username = :username
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {'username': session['username']}).fetchone()

    if row and all(r is not None for r in row):
        system_weights = dict(zip(ranking_attributes, row))
    else:
        try:
            with open('another_final_trained_weights.json', 'r') as f:
                system_weights = json.load(f)[-1]
        except:
            system_weights = {attr: 1.0 / len(ranking_attributes) for attr in ranking_attributes}

    # 合成 initial_weights
    alpha = 0.7
    initial_weights = {
        attr: alpha * user_weights[attr] + (1 - alpha) * system_weights.get(attr, 0)
        for attr in ranking_attributes
    }
    total_w = sum(initial_weights.values())
    initial_weights = {k: v / total_w for k, v in initial_weights.items()}

    # 评分并选择 top10
    scaler = MinMaxScaler()
    df = pd.DataFrame(filtered_products)
    df_original = df.copy()
    # 替换 review_counts 为 log1p(review_counts)，并独立归一化
    df_scaled = df.copy()
    df_scaled['review_counts'] = np.log1p(df['review_counts'])
    df_scaled['review_counts'] = scaler.fit_transform(df_scaled[['review_counts']])

    # 其余属性归一化
    for attr in ranking_attributes:
        if attr != 'review_counts':
            df_scaled[attr] = scaler.fit_transform(df[[attr]])

    # 对价格类反转归一化
    for attr in lower_is_better:
        df_scaled[attr] = 1.0 - df_scaled[attr]

    # 计算打分
    df_scaled['score'] = sum(df_scaled[attr] * initial_weights[attr] for attr in ranking_attributes)
    top10_indices = df_scaled.sort_values(by='score', ascending=False).head(10).index
    reorder_products = df_original.loc[top10_indices].to_dict(orient='records')

    return render_template('reorder.html', products=reorder_products)


# ---- 用户提交 reorder 排序并推荐 ----
@app.route('/submit_reorder', methods=['POST'])
@login_required
def submit_reorder():
    global reorder_products, initial_weights

    user_feedback_ranks = {}
    for i in range(1, len(reorder_products) + 1):
        rank = int(request.form.get(f'ranking_{i}'))
        user_feedback_ranks[reorder_products[i - 1]['product_name']] = rank

    df_feedback = pd.DataFrame(reorder_products)
    ranking_attributes = ['current_price', 'unit_price', 'star_rating', 'review_counts']
    lower_is_better = ['current_price', 'unit_price']

    # 替换 review_counts 为 log1p 并独立归一化
    scaler = MinMaxScaler()
    df_scaled = df_feedback.copy()
    df_scaled['review_counts'] = np.log1p(df_feedback['review_counts'])
    df_scaled['review_counts'] = scaler.fit_transform(df_scaled[['review_counts']])

    # 其余属性归一化
    for attr in ranking_attributes:
        if attr != 'review_counts':
            df_scaled[attr] = scaler.fit_transform(df_feedback[[attr]])

    # 对 lower_is_better 的特征反转
    for attr in lower_is_better:
        df_scaled[attr] = 1.0 - df_scaled[attr]

    # 生成 reorder feedback weights
    attribute_updates = {attr: 0.0 for attr in ranking_attributes}
    max_rank_feedback = len(user_feedback_ranks)

    for i, row in df_scaled.iterrows():
        product_name = df_feedback.loc[i, 'product_name']
        if product_name not in user_feedback_ranks:
            continue
        rank = user_feedback_ranks[product_name]
        preference = (max_rank_feedback - rank) / (max_rank_feedback - 1) if max_rank_feedback > 1 else 1.0
        for attr in ranking_attributes:
            attribute_updates[attr] += preference * row[attr]

    total_delta = sum(attribute_updates.values())
    reorder_weights = {k: v / total_delta for k, v in attribute_updates.items()} if total_delta > 0 else {
        k: 1.0 / len(attribute_updates) for k in ranking_attributes
    }

    # 计算最终 combined weights
    beta = 0.1
    combined_weights = {
        attr: (1 - beta) * initial_weights[attr] + beta * reorder_weights[attr] for attr in ranking_attributes
    }
    total_final = sum(combined_weights.values())
    combined_weights = {k: v / total_final for k, v in combined_weights.items()}

    # 保存最终权重
    update_query = text("""
        UPDATE user_accounts
        SET weight_star_rating = :star,
            weight_review_counts = :review,
            weight_current_price = :price,
            weight_unit_price = :unit
        WHERE username = :username
    """)
    with engine.begin() as conn:
        conn.execute(update_query, {
            'star': combined_weights['star_rating'],
            'review': combined_weights['review_counts'],
            'price': combined_weights['current_price'],
            'unit': combined_weights['unit_price'],
            'username': session['username']
        })

    # 推荐 top5
    final_scores = np.zeros(len(df_scaled))
    for attr in ranking_attributes:
        final_scores += df_scaled[attr] * combined_weights[attr]

    df_feedback['final_score'] = final_scores
    top5_df = df_feedback.sort_values(by='final_score', ascending=False).head(5)

    return render_template('recommend.html', products=top5_df.to_dict(orient='records'))

# ---- 启动服务器 ----
if __name__ == '__main__':
    app.run(debug=True)