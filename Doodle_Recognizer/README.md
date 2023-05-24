# Machine_Learning Project

* ***Juan Nicolás Sepúlveda Arias***
* ***Juan Nicolás Quintero Quintero***

El propósito de este proyecto es llevar a cabo un trabajo de clasificación sobre dibujos sencillos generados por el usuario mediante una interfaz gráfica. Para esto, haremos uso de redes neuronales convolucionales que trabajen sobre el input del usuario, leído como una matriz de intensidades en escala de grises.

Nuestro modelo clasfica los dibujos del usuario en 10 clases las cuales son:

* Reloj
* Boomerang
* Avión
* Caracol
* Paracaídas
* Árbol
* Pez
* Diamante
* Helicóptero
* Camiseta

Para este proyecto, en primera instancia se realizó un entrenamiento de la red, probando diferentes arquitecturas y formas de manejar los daros para ser capaces de entrenar la red con una mayor cantidad de datos. Posteriormente, el modelo generado se guarda y se carga para realizar las predicciones.

La tarea de clasificación está descrita en el archivo 10-Class_Optimized_CIFAR-10_With_GPU.ipynb, junto con la comparación con un modelo preeliminar de menos clases realizado con una red neuronal convencional.

Los demás archivos presentes en la carpeta del proyecto corresponden a la aplicación y a otros archivos de prueba. Si se desea utilizar la herramienta, los archivos necesarios son **board.py, predict.py** y **preprocess.py**. Para correrlo se debe ejecutar **board.py**. A continuación hay un par de imágenes que muestran el resultado obtenido.

![Pez.png](https://github.com/IamTrollFace555/ML_Project/blob/main/Fish.png)
![Tree.png](https://github.com/IamTrollFace555/ML_Project/blob/main/Tree.png)
