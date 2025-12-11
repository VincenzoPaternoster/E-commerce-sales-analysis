USE ecommerce;
BEGIN TRANSACTION;

BEGIN --  Customers data cleaning

--- country field has Null values 
--- I decided to replace NULL values with unknown (29 unknown)

UPDATE dbo.Customer
SET country='unknown'
WHERE country is NULL;

--- Then I apply lowercase on country field items
UPDATE dbo.Customer
SET country=LOWER(TRIM(country));

select * from Customer;

--- Search for a duplicate values
WITH dup AS (
    SELECT 
        customer_id,
        signup_date,
        first_name,
        last_name,
        email,
        ROW_NUMBER() OVER (
            PARTITION BY email, signup_date, first_name, last_name
            ORDER BY customer_id
        ) AS rn,
        COUNT(*) OVER (
            PARTITION BY email, signup_date, first_name, last_name
        ) AS cnt
    FROM Customer
)
SELECT * --- create temporary table 
INTO #temp
FROM dup
WHERE cnt > 1
ORDER BY email, signup_date, first_name, last_name, rn;

--- See which how many orders have the same email but different customer_id
SELECT t.customer_id,t.first_name,t.last_name,t.email,o.order_id,COUNT(o.order_date) as number_order,ROUND(SUM(o.total_amount),2) AS total 
FROM  #temp t
JOIN Orders o
ON t.customer_id=o.customer_id
WHERE cnt>1
GROUP BY t.customer_id,t.first_name,t.last_name,t.email,o.order_id;

--- Replace the customer_id of duplicate emails with the lowest customer_id among the different customer_id 
--- Search for a duplicate values
WITH dup AS (
    SELECT 
        customer_id,
        signup_date,
        first_name,
        last_name,
        email,
        ROW_NUMBER() OVER (
            PARTITION BY email, signup_date, first_name, last_name
            ORDER BY customer_id
        ) AS rn,
        COUNT(*) OVER (
            PARTITION BY email, signup_date, first_name, last_name
        ) AS cnt
    FROM Customer
)
SELECT 
       email,MIN(customer_id) AS survivor_id
INTO #id_surv
FROM dup
WHERE cnt>1
GROUP BY email;

--- Replace id of duplicate emails with lowest customer_id
--- Orders table

--- Create email column
ALTER TABLE Orders
ADD email VARCHAR(20);

--- Add email in email column
UPDATE Orders
SET Orders.email=c.email
FROM Customer c
WHERE Orders.customer_id=c.customer_id;

--- Replace customer_id of duplicate emails with lowest customer_id
UPDATE Orders 
SET customer_id=i.survivor_id
FROM #id_surv i
WHERE Orders.email=i.email;

--- Delete duplicate email per customer_id in Customer table
DELETE c
FROM Customer c
JOIN #id_surv i
ON c.email=i.email
WHERE c.customer_id<>i.survivor_id;


--- CAST signup_date from datetime(7) to date
UPDATE Customer
SET signup_date=CAST(signup_date AS DATE)

ALTER TABLE Customer
ALTER COLUMN signup_date DATE;

END



BEGIN -- Orders data cleaning

ALTER TABLE Orders
DROP COLUMN email;

--- Cast order_date
UPDATE Orders
SET order_date=CAST(order_date AS DATE);
ALTER TABLE Orders
ALTER COLUMN order_date DATE;


--- TRIM and LOWER CASE channel and status fields
UPDATE Orders
SET channel=LOWER(TRIM(channel));
UPDATE Orders
SET status=LOWER(TRIM(status));


--- Manage negative values in total amount field
select *
FROM Orders
WHERE total_amount<0;
--- There are 21 orders with a negative total amount. It seems those negative values are associated to status "cancelled" and "completed"

--- Product_id -> Order_id
--- Product_table -> Orders_items_table -> Orders
DROP TABLE IF EXISTS #newpric;
WITH new AS(
		SELECT oi.order_id,p.product_id,p.price,oi.quantity
		FROM Orders_items oi
		JOIN Products p
		ON oi.product_id=p.product_id
		)
SELECT new.order_id,new.product_id,new.price,new.quantity
INTO #newpric
FROM new
JOIN Orders o
ON new.order_id=o.order_id;
 
--- There are 3 Product_id (49,141,154) associated with negative prices. 
--- Since I do not know whether these anomalies are due to errors or other causes,
--- I will calculate a new field called ‘new_amount’ to replace the negative values with 0.

UPDATE #newpric
SET price=0
WHERE price<0; --- 146 orders contain three products with negative values

--- Calculate new_amount
ALTER TABLE #newpric
ADD new_amount DECIMAL(10,2);

WITH new_tot AS (select order_id,ROUND(SUM(price*quantity),2) AS new_amount from #newpric GROUP BY order_id)
UPDATE #newpric
SET new_amount=new_tot.new_amount
FROM new_tot
WHERE #newpric.order_id=new_tot.order_id;



--- Make join with Orders table
ALTER TABLE Orders
ADD new_amount DECIMAL(10,2);

UPDATE Orders 
SET Orders.new_amount=np.new_amount
FROM #newpric np
WHERE Orders.order_id=np.order_id; 

--- Manage order_date NULL values

--- Create new table with null values
DROP TABLE IF EXISTS date_anomalies;
SELECT *
INTO date_anomalies
FROM Orders
WHERE order_date IS NULL;

--- Create new table without null values
DROP TABLE IF EXISTS Orders2
SELECT *
INTO Orders2
FROM Orders
WHERE order_date IS NOT NULL;

END





BEGIN -- Orders_Items data cleaning
--- There are orders (20) with product with zero quantity 
--- I create new table "order_anomalies" with these products
DROP TABLE IF EXISTS order_anomalies;
SELECT order_id,product_id,quantity
INTO order_anomalies
FROM Orders_items
WHERE quantity=0;

--- I create new table "ordersitems2" withouth these products

DROP TABLE IF EXISTS ordersitems2;
SELECT order_id,product_id,quantity
INTO ordersitems2
FROM Orders_items
WHERE quantity>0;

END



BEGIN -- Products data cleaning

--- TRIM and LOWER CASE product_name field
UPDATE Products
SET product_name=LOWER(TRIM(product_name));

--- Fix grammar errors in categories name
SELECT DISTINCT(category),COUNT(category) FROM Products GROUP BY category;
---- Eletronics ->Electronics
---- Homes-> Home
UPDATE Products
SET category='Electronics'
WHERE category='Eletronics';

UPDATE Products
SET category='Home'
WHERE category='Homes';

--- TRIM and LOWER CASE category field
UPDATE Products
SET category=LOWER(TRIM(category));

-- ROUND price and creating a new column new_price with 0 instead of negative values  

ALTER TABLE Products
ADD new_price DECIMAL(10,2);

UPDATE Products
SET new_price=ROUND(price,2);

UPDATE Products
SET new_price=0
WHERE new_price<0;

END