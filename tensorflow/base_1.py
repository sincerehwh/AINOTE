
import tensorflow as tf 


val_1 = tf.constant([1.0,2.0],name="val_1")
val_2 = tf.constant([2.0,2.0],name="val_2")
add = tf.add(val_1,val_2,name="add_")

print(val_1)
print(val_2)
print(add)

