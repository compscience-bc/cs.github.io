const myFirstTensor = tf.scalar(42);
//console.log(myFirstTensor);
//myFirstTensor.print();

const oneDimTensor = tf.tensor1d([1, 2, 3]);
oneDimTensor.print();

// tampilkan di dokumen
// --> tidak tampil
//var inHTMLroot = document.getElementById('root').innerHTML;
//inHTMLroot += '<p> myFirstTensor:' + myFirstTensor + '<br> oneDimTensor:'  + oneDimTensor + '</p>'

/*document.getElementById('root').innerHTML += '<p>'
document.getElementById('root').innerHTML +=  'myFirstTensor:' + myFirstTensor + '<br>'
document.getElementById('root').innerHTML += 'oneDimTensor:'  + oneDimTensor
document.getElementById('root').innerHTML += '</p>'*/

document.getElementById('root').innerHTML += `<p>
  myFirstTensor: ${myFirstTensor.dataSync()}<br>
  oneDimTensor: ${oneDimTensor.dataSync()}
</p>`
	
// Preparing the training data
function fibonacci(num) {
  var a = 1,
    b = 0,
    temp;
  var seq = [];

  while (num > 0) {
    temp = a;
    a = a + b;
    b = temp;
    seq.push(b);
    num--;
  }

  return seq;
}

const fibs = fibonacci(10);

const xs = tf.tensor1d(fibs.slice(0, fibs.length - 1));
const ys = tf.tensor1d(fibs.slice(1));

const xmin = xs.min();
const xmax = xs.max();
const xrange = xmax.sub(xmin);

function norm(x) {
  return x.sub(xmin).div(xrange);
}

xsNorm = norm(xs);
ysNorm = norm(ys);

// Building our model

const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));

a.print();
b.print();

function predict(x) {
  return tf.tidy(() => {
	// Y = a.X  + b
    return a.mul(x).add(b);
  });
}

// Training

function loss(predictions, labels) {
  return predictions
    .sub(labels)
    .square()
    .mean();
}

const learningRate = 0.5;
const optimizer = tf.train.sgd(learningRate);

//const numIterations = 10000;
const numIterations = 100;
const errors = [];

for (let iter = 0; iter < numIterations; iter++) {
  optimizer.minimize(() => {
    const predsYs = predict(xsNorm);
    const e = loss(predsYs, ysNorm);
    errors.push(e.dataSync());
    return e;
  });
}

// Making predictions

console.log(errors[0]);
console.log(errors[numIterations - 1]);

//xTest = tf.tensor1d([2, 354224848179262000000]);
xTest = tf.tensor1d([2, 3, 4, 5]);
//xTest = tf.tensor1d([2, 5]);
predict(xTest).print();

document.getElementById('root').innerHTML += `<p>
  xTest = ${xTest.dataSync()} <br><br>
  Hasil Prediksi = ${predict(xTest).dataSync()}
</p>`

//a.print();
//b.print();

document.getElementById('root').innerHTML += `<p>
  Y= ${a.dataSync()}X + ${b.dataSync()}
</p>`

