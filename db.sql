-- Cities Table
CREATE TABLE cities (
    city_id INT PRIMARY KEY AUTO_INCREMENT,
    city_name VARCHAR(255) NOT NULL
);

-- Places Table with Image
CREATE TABLE places (
    place_id INT PRIMARY KEY AUTO_INCREMENT,
    place_name varchar(255),
    city_id INT,
    description TEXT,
    type VARCHAR(50),
    rating DECIMAL(3, 2) DEFAULT 0.0,
    image_url VARCHAR(255), -- You can adjust the size based on your needs
    FOREIGN KEY (city_id) REFERENCES cities(city_id)
);

-- Reviews Table (unchanged)
CREATE TABLE reviews (
    review_id INT PRIMARY KEY AUTO_INCREMENT,
    place_id INT,
    user_id INT,
    rating DECIMAL(2, 1),
    comment_text TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (place_id) REFERENCES places(place_id)
);
CREATE TABLE visited_places (
    visit_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    place_id INT,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (place_id) REFERENCES places(place_id)
);

-- Insert data into Cities table
INSERT INTO cities (city_name) VALUES
    ('Casablanca'),
    ('Rabat'),
    ('Fez');

-- Insert data into Places table for Morocco with image paths starting from /static
INSERT INTO places (place_name, city_id, description, type, rating, image_url)
VALUES
    ('Hassan II Mosque', 1, 'One of the largest mosques in the world located in Casablanca.', 'Religious', 4.9, '/static/img_maroc/Hassan II Mosque.jpeg'),
    ('King\'s Palace', 2, 'The official residence of the King of Morocco in Rabat.', 'Historical', 4.6, '/static/img_maroc/King\'s Palace.jpeg'),
    ('La Sqala', 1, 'A historic fortress turned restaurant in Casablanca.', 'Culinary', 4.4, '/static/img_maroc/La Sqala.jpeg'),
    ('Old Medina', 3, 'The ancient walled city of Fez with narrow winding streets.', 'Historical', 4.7, '/static/img_maroc/Old Medina.jpeg'),
    ('Moroccan Jewish Museum', 1, 'A museum in Casablanca showcasing the history of Moroccan Jews.', 'Cultural', 4.5, '/static/img_maroc/Moroccan Jewish Museum.jpeg'),
    ('Parc de la Ligue Arabe', 1, 'A popular park in Casablanca with green spaces and gardens.', 'Recreational', 4.2, '/static/img_maroc/Parc de la Ligue Arabe.jpeg'),
    ('Royal Golf Anfa Mohammedia', 1, 'A prestigious golf course located in Casablanca.', 'Recreational', 4.8, '/static/img_maroc/Royal Golf Anfa Mohammedia.jpeg'),
    ('The Corniche', 1, 'A scenic coastal road along the Atlantic Ocean in Casablanca.', 'Scenic', 4.6, '/static/img_maroc/The Corniche.jpeg'),
    ('Villa des Arts', 1, 'An art museum in Casablanca featuring contemporary Moroccan art.', 'Cultural', 4.3, '/static/img_maroc/Villa des Arts.jpeg');

-- Insert data into Reviews table for Morocco places
INSERT INTO reviews (place_id, user_id, rating, comment_text, timestamp)
VALUES
    (1, 1, 4.5, 'Magical experience at Hassan II Mosque!', '2023-01-01 12:00:00'),
    (2, 2, 4.0, 'Impressive architecture at King\'s Palace.', '2023-01-02 14:30:00'),
    (3, 1, 4.8, 'Delicious food and historic ambiance at La Sqala.', '2023-01-03 10:45:00'),
    (4, 2, 5.0, 'Exploring the ancient charm of Old Medina.', '2023-01-04 18:20:00'),
    (5, 1, 4.5, 'Informative visit to the Moroccan Jewish Museum.', '2023-01-05 09:30:00'),
    (6, 2, 4.2, 'Relaxing day at Parc de la Ligue Arabe.', '2023-01-06 15:15:00'),
    (7, 1, 4.8, 'Enjoyed a round of golf at Royal Golf Anfa Mohammedia.', '2023-01-07 11:00:00'),
    (8, 2, 4.6, 'Scenic views along The Corniche.', '2023-01-08 16:45:00'),
    (9, 1, 4.3, 'Artistic inspiration at Villa des Arts.', '2023-01-09 13:30:00');
