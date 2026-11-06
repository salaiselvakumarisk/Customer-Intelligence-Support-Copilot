
-- QUERY: Q1
-- Customers with more than INR 8,000 spend on orders above INR 500
SELECT
    customer_id,
    ROUND(SUM(order_amount), 2) AS total_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY total_spend DESC;


-- QUERY: Q2
-- Revenue by city and product category
SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- QUERY: Q3
-- Products priced above the average catalogue price
SELECT
    product_id,
    product_name,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;


-- QUERY: Q4
-- Customers without any completed order
SELECT
    c.customer_id,
    c.city
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- QUERY: Q5
-- Customers whose completed-order spending is above average
WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
)
SELECT
    customer_id,
    ROUND(total_spend, 2) AS total_spend
FROM customer_spend
WHERE total_spend > (
    SELECT AVG(total_spend) FROM customer_spend
)
ORDER BY total_spend DESC;


-- QUERY: Q6
-- Ranking customers by completed-order spend within each city
WITH customer_spend AS (
    SELECT
        c.city,
        c.customer_id,
        SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT
    city,
    customer_id,
    ROUND(total_spend, 2) AS total_spend,
    RANK() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY city ORDER BY total_spend DESC
    ) AS dense_spend_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city ORDER BY total_spend DESC, customer_id
    ) AS row_number
FROM customer_spend
ORDER BY city, spend_rank, customer_id;


-- QUERY: Q7
-- Daily completed revenue and cumulative revenue
WITH daily_revenue AS (
    SELECT
        substr(order_ts, 1, 10) AS order_day,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY substr(order_ts, 1, 10)
)
SELECT
    order_day,
    ROUND(daily_revenue, 2) AS daily_revenue,
    ROUND(
        SUM(daily_revenue) OVER (
            ORDER BY order_day
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ), 2
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY order_day;
