import tensorflow as tf

model = tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(64,(3,3), activation='relu', input_shape=(180, 180, 3)), 
   tf.keras.layers.MaxPooling2D(2,2),   
   tf.keras.layers.Conv2D(64,(3, 3), activation='relu'),
   tf.keras.layers.MaxPooling2D(2,2),  
   tf.keras.layers.Conv2D(128,(3, 3), activation='relu'),
   tf.keras.layers.MaxPooling2D(2,2),
   tf.keras.layers.Conv2D(128,(3, 3), activation='relu'),
   tf.keras.layers.MaxPooling2D(2,2),    

   tf.keras.layers.Flatten(),
   tf.keras.layers.Dropout(0.5),

   tf.keras.layers.Dense(512, activation='relu'),
   tf.keras.layers.Dense(2, activation='softmax')
])


