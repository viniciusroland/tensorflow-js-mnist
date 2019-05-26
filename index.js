//importando tensorflow.js para node
const tf = require('@tensorflow/tfjs-node')

//importando dataset de imagens de digitos
const mnist = require('mnist')

//setando 10000 dados para treino, 2000 para teste
const set = mnist.set(10000, 2000)

//separando treino e teste
const training_set = set.training
const test_set = set.test

//cada set tem um input (length == 784) e output (length == 10)

//separando inputs e outputs de treino
let training_set_inputs = []
let training_set_outputs = []
for(let element of training_set) {
  training_set_inputs.push(element.input)
  training_set_outputs.push(element.output)
}

//separando inputs e outputs de teste
let test_set_inputs = []
let test_set_outputs = []
for(let element of test_set) {
  test_set_inputs.push(element.input)
  test_set_outputs.push(element.output)
}

//criando modelo de rede neural
const model = tf.sequential()

//adicionando a primeira hidden layer
model.add(tf.layers.dense({
  units : 32, // 32 unidades (neuronios)
  activation : 'relu', // funcao de ativação : relu
  inputShape : [784] // recebendo 784 inputs (todos os pixels normalizados de uma imagem)
}))


//layer de output
model.add(tf.layers.dense({
  units : 10, // vetor de resultado; ex: numero 2 == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] (corresponde ao index do elemento 1 no vetor)
  activation : 'softmax', //funcao de ativação : softmax
}))

//setando a velocidade de crescimento e compilando a rede neural
const learning_rate = 0.1
const optimizer = tf.train.sgd(learning_rate)
model.compile({
  optimizer : optimizer,
  loss: tf.losses.meanSquaredError
})

//montando Tensor de input de treino
const xs = tf.tensor2d(
  training_set_inputs
)

//monstando Tensor de output de treino
const ys = tf.tensor2d(
  training_set_outputs
)


//treinando o modelo
train_model().then(() => {
  //testando o modelo com alguns digitos 0-9
  testar(1)
  testar(4)
  testar(7)
  testar(0)
  testar(5)
  testar(9)
  testar(2)
  testar(3)
})

//funcao de treino para a rede neural
async function train_model() {
  for(let i = 0; i < 10; i++) {
    //i -> 10 e epochs -> 10, ou seja, treina 100 vezes
    // fita o modelo com os inputs e outputs de treino com 10 epocas e embaralhando os elementos a cada ciclo
    const response = await model.fit(xs, ys, {epochs : 10, shuffle: true})
  }
  //limpando a memoria
  xs.dispose()
  ys.dispose()

}

//funcao basica de teste
function testar(n) {
  //pega o vetor de pixels (784) de um digito n
  const test_mnist = mnist[n].get()

  //transforma o vetor num Tensor
  const tensor_test = tf.tensor2d([test_mnist])

  //rede neural atua
  const result = model.predict(tensor_test)

  //retorna um resultado (ex: [0.001, 0.003232, 0.00091, 0.9889, 0.0001, 0, 0, 0.0982, 0.001])
  //nesse caso o digito é o 3
  result.print()
  console.log('> Teste finalizado!')

  //limpa a memoria
  tensor_test.dispose()
  
}
