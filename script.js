console.log('Iniciando definición de classNames');  
const classNames = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];  
console.log('classNames definido:', classNames);  

let model;  
let webcam; // Esta variable almacenará la instancia de la cámara  

// Crear el modelo  
async function createModel() {  
    model = tf.sequential();  
    model.add(tf.layers.flatten({ inputShape: [224, 224, 3] })); // Asumiendo que la entrada es de 224x224 píxeles en color  
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));  
    model.add(tf.layers.dense({ units: classNames.length, activation: 'softmax' })); // Número de clases en classNames  

    model.compile({  
        optimizer: 'adam',  
        loss: 'categoricalCrossentropy',  
        metrics: ['accuracy'],  
    });  
}  

// Cargar el modelo  
async function loadModel() {  
    const modelPath = 'http://localhost:8000/modelo/model.json'; // Cambia a la ruta correcta  
    model = await tf.loadLayersModel(modelPath);  
    console.log('Modelo cargado');  
}  

// Iniciar la cámara   
async function startWebcam() {  
    const videoElement = document.getElementById('webcam');  
    try {  
        webcam = await tf.data.webcam(videoElement);  
        console.log('Webcam iniciada:', webcam);  
        return webcam;  
    } catch (error) {  
        console.error('Error al iniciar la cámara:', error);  
        return null; // Manejo de error  
    }  
}  

// Capturar imagen desde la cámara  
async function capture(webcam) {  
    try {  
        const img = await webcam.capture();  
        return img;   
    } catch (error) {  
        console.error('Error al capturar la imagen:', error);  
        return null; // Manejo del error  
    }  
}  

// Predecir animal  
async function predictAnimal() {  
    

    const resizedImg = tf.image.resizeBilinear(img, [224, 224]); // Cambia el tamaño a 224x224  
    const normalizedImg = resizedImg.div(255.0); // Normaliza los valores entre 0 y 1  
    const reshapedImg = normalizedImg.expandDims(0); // Asegúrate de que tenga la forma [1, 224, 224, 3]  

    // Realiza la predicción  
    const prediction = model.predict(reshapedImg);  
    const predictedClass = classNames[prediction.argMax(-1).dataSync()[0]];  

    // Mostrar el resultado  
    document.getElementById('prediction-result').innerText = `Predicción: ${predictedClass}`;  
    
    // Limpiar memoria  
    img.dispose();  
    resizedImg.dispose();  
    normalizedImg.dispose();  
    reshapedImg.dispose();  
}  

// Inicializar el modelo y la cámara cuando la página se cargue  
async function init() {  
    await createModel(); // Asegúrate de que el modelo se cree primero  
    await loadModel();   // Luego carga el modelo  
    await startWebcam(); // Después inicia la cámara  
}  
// Configurar el evento de predicción  
document.getElementById('predict-button').addEventListener('click', async () => {  
    await predictAnimal();  
});  

// Llamar a la función init al cargar el script  
init();



