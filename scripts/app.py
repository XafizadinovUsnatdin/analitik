from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
import pandas as pd
import json
import os
from datetime import datetime, timedelta, date
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from models import db, User, OrderSettings, Order
from werkzeug.security import generate_password_hash
import secrets
from functools import wraps
from dotenv import load_dotenv
import gunicorn

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///../data/app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Premium required decorator
def premium_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_premium:
            flash('Bu funksiya faqat premium foydalanuvchilar uchun mavjud.', 'warning')
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Create database tables
@app.before_first_request
def create_tables():
    db.create_all()
    # Create admin user if not exists
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@example.com', is_premium=True)
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()

# Load data
def load_data():
    products_df = pd.read_csv('data/products_with_shelf_life.csv')
    stock_df = pd.read_csv('data/stock_data.csv')
    sales_df = pd.read_csv('data/sales_expanded_1000.csv')
    
    # Convert date to datetime
    sales_df['date'] = pd.to_datetime(sales_df['date'], format='%d/%m/%Y')
    
    # Load forecasts
    forecasts = {}
    forecast_dir = 'data/forecasts'
    for file in os.listdir(forecast_dir):
        if file.startswith('forecast_') and file.endswith('.json'):
            with open(os.path.join(forecast_dir, file), 'r') as f:
                month = file.split('_')[1].split('.')[0]
                forecasts[month] = json.load(f)
    
    return products_df, stock_df, sales_df, forecasts

# Calculate orders
def calculate_orders(product_id, forecast_data, stock_data, product_data, days_to_maintain=7, min_order_quantity=10):
    # Get current stock
    current_stock = stock_data[stock_data['product_id'] == product_id]['stock'].values[0]
    
    # Get shelf life
    shelf_life = product_data[product_data['id'] == product_id]['shelf_life_days'].values[0]
    
    # Sort forecast data by date
    forecast_data = sorted(forecast_data, key=lambda x: x['date'])
    
    # Calculate cumulative demand
    cumulative_demand = 0
    orders = []
    
    for i, day in enumerate(forecast_data):
        date_obj = datetime.strptime(day['date'], '%Y-%m-%d').date()
        daily_demand = day['predicted_quantity']
        cumulative_demand += daily_demand
        
        # If stock will run out in the next days_to_maintain days
        if current_stock - cumulative_demand < daily_demand * days_to_maintain:
            # Calculate order quantity (enough for shelf_life days or at least min_order_quantity)
            days_to_order = min(shelf_life, 30)  # Don't order more than 30 days worth
            future_demand = sum([d['predicted_quantity'] for d in forecast_data[i:i+days_to_order] if i+days_to_order < len(forecast_data)])
            order_quantity = max(future_demand, min_order_quantity)
            
            orders.append({
                'product_id': product_id,
                'suggested_date': date_obj,
                'quantity': round(order_quantity, 2),
                'days_covered': days_to_order
            })
            
            # Update stock after order
            current_stock += order_quantity
            cumulative_demand = 0
    
    return orders

# Routes
@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('dashboard'))
        else:
            flash('Login yoki parol noto\'g\'ri', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Bu foydalanuvchi nomi allaqachon mavjud', 'danger')
            return redirect(url_for('register'))
            
        if User.query.filter_by(email=email).first():
            flash('Bu email allaqachon ro\'yxatdan o\'tgan', 'danger')
            return redirect(url_for('register'))
        
        user = User(username=username, email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        flash('Ro\'yxatdan o\'tish muvaffaqiyatli yakunlandi', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    products_df, stock_df, _, _ = load_data()
    
    # Merge product and stock data
    merged_df = pd.merge(products_df, stock_df, left_on='id', right_on='product_id')
    
    # Calculate days of stock left
    merged_df['days_left'] = merged_df['days_to_cover']
    
    # Identify products with low stock
    merged_df['status'] = 'normal'
    merged_df.loc[merged_df['days_left'] <= 3, 'status'] = 'critical'
    merged_df.loc[(merged_df['days_left'] > 3) & (merged_df['days_left'] <= 7), 'status'] = 'warning'
    
    # Convert to list of dicts for template
    products = merged_df.to_dict('records')
    
    return render_template('dashboard.html', 
                          products=products, 
                          is_premium=current_user.is_premium)

@app.route('/product/<int:product_id>')
@login_required
def product_detail(product_id):
    products_df, stock_df, _, _ = load_data()
    product = products_df[products_df['id'] == product_id].iloc[0].to_dict() if not products_df[products_df['id'] == product_id].empty else None
    
    if not product:
        flash('Mahsulot topilmadi', 'danger')
        return redirect(url_for('dashboard'))
    
    # Get stock info
    stock = stock_df[stock_df['product_id'] == product_id].iloc[0].to_dict() if not stock_df[stock_df['product_id'] == product_id].empty else None
    
    # Get order settings
    order_settings = OrderSettings.query.filter_by(user_id=current_user.id, product_id=product_id).first()
    if not order_settings:
        order_settings = OrderSettings(
            user_id=current_user.id,
            product_id=product_id,
            days_to_maintain=7,
            min_order_quantity=10.0
        )
        db.session.add(order_settings)
        db.session.commit()
    
    return render_template('product_detail.html', 
                          product=product, 
                          stock=stock, 
                          order_settings=order_settings,
                          is_premium=current_user.is_premium)

@app.route('/orders')
@login_required
def orders():
    products_df, stock_df, _, forecasts = load_data()
    
    # Get all products
    products = products_df.to_dict('records')
    
    # Get existing orders
    user_orders = Order.query.filter_by(user_id=current_user.id).all()
    
    return render_template('orders.html', 
                          products=products, 
                          orders=user_orders,
                          is_premium=current_user.is_premium)

@app.route('/premium')
@login_required
def premium():
    return render_template('premium.html', is_premium=current_user.is_premium)

@app.route('/activate_premium', methods=['POST'])
@login_required
def activate_premium():
    if current_user.is_premium:
        flash('Sizda allaqachon premium obuna mavjud', 'info')
        return redirect(url_for('dashboard'))
    
    # In a real app, you would process payment here
    # For demo purposes, we'll just activate premium
    current_user.is_premium = True
    current_user.premium_until = datetime.utcnow() + timedelta(days=30)
    db.session.commit()
    
    flash('Premium obuna faollashtirildi!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/update_order_settings', methods=['POST'])
@login_required
def update_order_settings():
    product_id = request.form.get('product_id', type=int)
    days_to_maintain = request.form.get('days_to_maintain', type=int)
    min_order_quantity = request.form.get('min_order_quantity', type=float)
    
    if not product_id:
        flash('Mahsulot ID si ko\'rsatilmagan', 'danger')
        return redirect(url_for('dashboard'))
    
    order_settings = OrderSettings.query.filter_by(user_id=current_user.id, product_id=product_id).first()
    
    if not order_settings:
        order_settings = OrderSettings(
            user_id=current_user.id,
            product_id=product_id
        )
        db.session.add(order_settings)
    
    order_settings.days_to_maintain = days_to_maintain
    order_settings.min_order_quantity = min_order_quantity
    db.session.commit()
    
    flash('Buyurtma sozlamalari yangilandi', 'success')
    return redirect(url_for('product_detail', product_id=product_id))

@app.route('/calculate_orders/<int:product_id>')
@login_required
def calculate_product_orders(product_id):
    products_df, stock_df, _, forecasts = load_data()
    
    # Get product info
    product = products_df[products_df['id'] == product_id].iloc[0].to_dict() if not products_df[products_df['id'] == product_id].empty else None
    
    if not product:
        return jsonify({"error": "Mahsulot topilmadi"}), 404
    
    # Get order settings
    order_settings = OrderSettings.query.filter_by(user_id=current_user.id, product_id=product_id).first()
    if not order_settings:
        order_settings = OrderSettings(
            user_id=current_user.id,
            product_id=product_id
        )
        db.session.add(order_settings)
        db.session.commit()
    
    # Get forecast data
    forecast_data = []
    for month in ['2025-01', '2025-02', '2025-03']:
        if month in forecasts and str(product_id) in forecasts[month]:
            forecast_data.extend(forecasts[month][str(product_id)]['forecast'])
    
    # Calculate orders
    orders = calculate_orders(
        product_id, 
        forecast_data, 
        stock_df, 
        products_df,
        days_to_maintain=order_settings.days_to_maintain,
        min_order_quantity=order_settings.min_order_quantity
    )
    
    # Save orders to database
    for order_data in orders:
        # Check if order already exists for this date and product
        existing_order = Order.query.filter_by(
            user_id=current_user.id,
            product_id=product_id,
            suggested_date=order_data['suggested_date']
        ).first()
        
        if not existing_order:
            new_order = Order(
                user_id=current_user.id,
                product_id=product_id,
                quantity=order_data['quantity'],
                suggested_date=order_data['suggested_date'],
                status='pending'
            )
            db.session.add(new_order)
    
    db.session.commit()
    
    flash('Buyurtmalar hisoblandi va saqlandi', 'success')
    return redirect(url_for('orders'))

@app.route('/update_order_status/<int:order_id>/<status>')
@login_required
def update_order_status(order_id, status):
    if status not in ['pending', 'completed', 'cancelled']:
        flash('Noto\'g\'ri status', 'danger')
        return redirect(url_for('orders'))
    
    order = Order.query.filter_by(id=order_id, user_id=current_user.id).first()
    
    if not order:
        flash('Buyurtma topilmadi', 'danger')
        return redirect(url_for('orders'))
    
    order.status = status
    db.session.commit()
    
    flash('Buyurtma statusi yangilandi', 'success')
    return redirect(url_for('orders'))

# API Routes
@app.route('/api/product/<int:product_id>')
@login_required
def get_product_data(product_id):
    products_df, stock_df, sales_df, forecasts = load_data()
    
    # Get product info
    product = products_df[products_df['id'] == product_id].iloc[0].to_dict() if not products_df[products_df['id'] == product_id].empty else None
    
    if not product:
        return jsonify({"error": "Product not found"}), 404
    
    # Get stock info
    stock = stock_df[stock_df['product_id'] == product_id].iloc[0].to_dict() if not stock_df[stock_df['product_id'] == product_id].empty else None
    
    # Get historical sales (last 90 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    historical_sales = sales_df[
        (sales_df['product_id'] == product_id) & 
        (sales_df['date'] >= start_date) & 
        (sales_df['date'] <= end_date)
    ]
    
    historical_data = []
    for _, row in historical_sales.iterrows():
        historical_data.append({
            "date": row['date'].strftime('%Y-%m-%d'),
            "quantity": float(row['quantity'])
        })
    
    # Get forecasts for next 3 months
    forecast_data = []
    for month in ['2025-01', '2025-02', '2025-03']:
        if month in forecasts and str(product_id) in forecasts[month]:
            forecast_data.extend(forecasts[month][str(product_id)]['forecast'])
    
    # Filter forecast data based on period
    period = request.args.get('period', 'all')
    if period != 'all':
        today = date.today()
        
        if period == 'week':
            end_date = today + timedelta(days=7)
            forecast_data = [d for d in forecast_data if datetime.strptime(d['date'], '%Y-%m-%d').date() <= end_date]
        elif period == 'month':
            # Get the end of current month
            if today.month == 12:
                end_date = date(today.year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(today.year, today.month + 1, 1) - timedelta(days=1)
            forecast_data = [d for d in forecast_data if datetime.strptime(d['date'], '%Y-%m-%d').date() <= end_date]
        elif period == 'quarter':
            # Get the end of current quarter
            quarter = (today.month - 1) // 3 + 1
            if quarter == 4:
                end_date = date(today.year + 1, 1, 1) - timedelta(days=1)
            else:
                end_date = date(today.year, quarter * 3 + 1, 1) - timedelta(days=1)
            forecast_data = [d for d in forecast_data if datetime.strptime(d['date'], '%Y-%m-%d').date() <= end_date]
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    
    # Plot historical data
    if historical_data:
        dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in historical_data]
        quantities = [item['quantity'] for item in historical_data]
        plt.plot(dates, quantities, label='Tarixiy savdo', color='blue')
    
    # Plot forecast data
    if forecast_data:
        dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in forecast_data]
        quantities = [item['predicted_quantity'] for item in forecast_data]
        plt.plot(dates, quantities, label='Bashorat', color='red', linestyle='--')
    
    plt.title(f'Savdo tarixi va bashorati: {product["name"]}')
    plt.xlabel('Sana')
    plt.ylabel('Miqdor')
    plt.legend()
    plt.grid(True)
    
    # Save plot to base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Calculate orders
    order_settings = OrderSettings.query.filter_by(user_id=current_user.id, product_id=product_id).first()
    if not order_settings:
        order_settings = OrderSettings(
            user_id=current_user.id,
            product_id=product_id
        )
        db.session.add(order_settings)
        db.session.commit()
    
    orders = calculate_orders(
        product_id, 
        forecast_data, 
        stock_df, 
        products_df,
        days_to_maintain=order_settings.days_to_maintain,
        min_order_quantity=order_settings.min_order_quantity
    )
    
    return jsonify({
        "product": product,
        "stock": stock,
        "historical_data": historical_data,
        "forecast_data": forecast_data,
        "plot": plot_data,
        "orders": orders
    })

@app.route('/api/categories')
@login_required
def get_categories():
    products_df, _, _, _ = load_data()
    categories = products_df['category'].unique().tolist()
    return jsonify(categories)

@app.route('/api/products_by_category')
@login_required
def get_products_by_category():
    products_df, stock_df, _, _ = load_data()
    
    # Group products by category
    category_products = {}
    for category in products_df['category'].unique():
        products = products_df[products_df['category'] == category]
        
        # Merge with stock data
        merged = pd.merge(products, stock_df, left_on='id', right_on='product_id', how='left')
        
        category_products[category] = merged.to_dict('records')
    
    return jsonify(category_products)

if __name__ == '__main__':
    app.run(debug=True)

