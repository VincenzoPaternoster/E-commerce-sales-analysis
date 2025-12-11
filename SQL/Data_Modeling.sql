USE ecommerce;
BEGIN TRANSACTION

BEGIN -- Create dimension tables

--- Create Channel table
DROP TABLE IF EXISTS dim_chan;
CREATE TABLE dim_chan(
		channel_id INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
		chan_type VARCHAR(20) NOT NULL,
		);

INSERT INTO dim_chan (chan_type)
SELECT DISTINCT(channel) FROM Orders2;

--- Create Status table
DROP TABLE IF EXISTS dim_stat;
CREATE TABLE dim_stat(
		status_id INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
		stat_type VARCHAR(20) NOT NULL,
		);

INSERT INTO dim_stat (stat_type)
SELECT DISTINCT(status) FROM Orders2;

--- Create Date table
DROP TABLE IF EXISTS dim_date;
CREATE TABLE dim_date(
		date_id INT IDENTITY (1,1) NOT NULL PRIMARY KEY,
		date_og DATE NOT NULL,
		nb_year INT NOT NULL,
		quarter_id VARCHAR(10) NOT NULL,
		nb_month INT NOT NULL,
		nm_month VARCHAR(15) NOT NULL,
		nb_day INT NOT NULL,
		nm_day VARCHAR(15) NOT NULL,
		nb_week INT NOT NULL,
		season VARCHAR(10)		
		)

INSERT INTO dim_date(date_og,nb_year, quarter_id, nb_month, nm_month, nb_day, nm_day,nb_week, season)
SELECT 
	order_date AS date_og,
    YEAR(order_date) AS nb_year,
    CONCAT(YEAR(order_date), '-Q', DATEPART(QUARTER, order_date)) AS quarter_id,
    MONTH(order_date) AS nb_month,
    DATENAME(MONTH, order_date) AS nm_month,
    DAY(order_date) AS nb_day,
    DATENAME(WEEKDAY, order_date) AS nm_day,
	DATEPART(WEEK, order_date) AS nb_week,
    CASE
        WHEN MONTH(order_date) IN (12, 1, 2) THEN 'winter'
        WHEN MONTH(order_date) IN (3, 4, 5) THEN 'spring'
        WHEN MONTH(order_date) IN (6, 7, 8) THEN 'summer'
        ELSE 'autumn'
    END AS season
FROM (SELECT DISTINCT(order_date) FROM Orders2) AS dist_date;

--- Create season_id
DROP TABLE IF EXISTS dim_season;
CREATE TABLE dim_season (
		season_id INT NOT NULL IDENTITY (1,1) PRIMARY KEY,
		type_seas VARCHAR(10))
INSERT INTO dim_season
SELECT DISTINCT(season) FROM dim_date;

--- Create category table
DROP TABLE IF EXISTS dim_cat;
CREATE TABLE dim_cat (
		category_id INT NOT NULL IDENTITY(1,1) PRIMARY KEY,
		cat_type VARCHAR(15)
		)
INSERT INTO dim_cat
SELECT DISTINCT(category) FROM Products; 

--- Add category_id field in Products table
ALTER TABLE Products
ADD category_id INT;
UPDATE Products
SET category_id=dc.category_id
FROM dim_cat dc
WHERE Products.category=dc.cat_type;

END


BEGIN --- Create fact table

DROP TABLE IF EXISTS fact_sales;
CREATE TABLE fact_sales (
    order_id INT,
    customer_id INT,
    date_id INT,
    channel_id INT,
    status_id INT,
    product_id INT,
	product_name VARCHAR(15),
    category_id INT,
    quantity INT,
    price DECIMAL(10,2),
    amount DECIMAL(10,2)
);
INSERT INTO fact_sales (
    order_id, customer_id, date_id, channel_id, status_id,
    product_id,p.product_name, category_id, quantity, price, amount
)
SELECT 
    o.order_id,
    o.customer_id,
    d.date_id,
    ch.channel_id,
    st.status_id,
    p.product_id,
	p.product_name,
    c.category_id,
    oi.quantity,
    p.new_price,
    ROUND(oi.quantity * p.new_price, 2) AS amount
FROM Orders2 o
JOIN ordersitems2 oi ON o.order_id = oi.order_id
JOIN Products p ON oi.product_id = p.product_id
JOIN dim_cat c ON p.category = c.cat_type
JOIN dim_date d ON o.order_date = d.date_og
JOIN dim_chan ch ON o.channel = ch.chan_type
JOIN dim_stat st ON o.status = st.stat_type;

END


