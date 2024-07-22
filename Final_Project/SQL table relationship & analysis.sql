-- create db --
stock_price_db

-- import data from csv file into database --
using "table data import wizard"

-- creating date and time table --
CREATE TABLE date_table (
    date_id INT AUTO_INCREMENT PRIMARY KEY,
    date DATE UNIQUE
);
CREATE TABLE time_table (
    time_id INT AUTO_INCREMENT PRIMARY KEY,
    time TIME UNIQUE
);
CREATE TABLE stock_data (
    date_id INT,
    time_id INT,
    stock VARCHAR(45),
    open DECIMAL(10,4),
    high DECIMAL(10,4),
    low DECIMAL(10,4),
    close DECIMAL(10,4),
    volume INT,
    FOREIGN KEY(date_id) REFERENCES date_table(date_id),
    FOREIGN KEY(date_id) REFERENCES date_table(date_id)
);

-- Insert record into tables --
INSERT INTO date_table (date)
SELECT DISTINCT Date
FROM crowdstrike_data
ON DUPLICATE KEY UPDATE date = date_table.date;

INSERT INTO time_table (time)
SELECT DISTINCT Time
FROM crowdstrike_data
ON DUPLICATE KEY UPDATE time = time_table.time;

INSERT INTO stock_data (date_id, time_id, stock, open, high, low, close, volume)
SELECT d.date_id, t.time_id, 'CRWD', trans.open, trans.high, trans.low, trans.close, trans.volume
FROM crowdstrike_data trans
JOIN date_table d ON d.date = trans.Date
JOIN time_table t ON t.time = trans.Time;

-- EVENT ANALYSING QUERY --
--Sum of Volume--
    Volume:
    SUM (07/08, 07/09, 07/10)

-- if volume increase, it can indicate increase on investor interest
-- and trading activity based on the posistive news.

SELECT d.date, SUM(s.volume) AS total_volume
FROM stock_data s
JOIN date_table d ON s.date_id = d.date_id
WHERE d.date IN ('2024-07-08', '2024-07-09', '2024-07-10')
GROUP BY d.date
ORDER BY d.date;

OUTPUT:
        volume went from 3213858 to 6170052.
        Essentially it doubled, increased by approximately 92% from 3,213,858 to 6,170,052.


Month 07/01 - 07/19:

SELECT d.date, SUM(s.volume) AS total_volume
FROM stock_data s
JOIN date_table d ON s.date_id = d.date_id
WHERE d.date BETWEEN '2024-07-01' AND '2024-07-19'
GROUP BY d.date
ORDER BY d.date;
