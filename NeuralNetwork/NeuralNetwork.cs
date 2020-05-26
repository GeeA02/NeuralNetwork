using System;
using System.IO;
using System.Linq;

namespace NeuralNetwork
{
    class NeuralNetwork
    {
        Layer[] layers;
        double[] inputs;
        double previousDelta;
        double delta = 0;
        int deltaDiffCount = 0; 
        Random random = new Random();


        //layer.Lenght == count of layers and every item in array == count of neurons 
        public NeuralNetwork(int[] layer)
        {
            layers = new Layer[layer.Length];
            for (int i = 0; i < layers.Length - 1; i++)
                layers[i] = new Layer(layer[i], layer[i + 1], random); 
            //Last layer
            layers[layers.Length - 1] = new Layer(layer[layer.Length - 1], layer[layer.Length - 1], random);

        }

        public void Learning(double[] inputs, double[] expected)
        {
            this.inputs = inputs;
            FeedForward();
            //zmiana wag w przypadku gdy sieć wpadnie w lokalne minimum
            previousDelta = delta;
            delta = layers[layers.Length - 1].Neurons.Zip(expected, (n, e) => Math.Pow(n.Output - e, 2)).Sum();
            deltaDifference(previousDelta, delta, 5, 0.001);

            BackProp(expected);
            FeedForward();
            //ShowOutput(expected);
        }

        void BackProp(double[] expected)
        {
            //backpropagation for last layer
            layers[layers.Length - 1].BackPropOutput(expected, ref layers[layers.Length - 2]);
            // run over all layers backwards
            for (int i = layers.Length - 2; i > 0; i--)
                    layers[i].BackPropHidden(ref layers[i - 1], layers[i+1]); //back prop hidden

            foreach (Layer l in layers)
            {
                //Clear errors and gammas
                for (int i = 0; i < l.Gamma.Length; i++)
                {
                    l.Gamma[i] = 0;
                    l.Error[i] = 0;
                }
            }
        }

        void deltaDifference(double pDelta, double Delta, int count, double accuracy)
        {
            if (Math.Abs(pDelta - Delta) > accuracy)
            {
                if (deltaDiffCount == count)
                {
                    //liczba z przedziału <-0.05, 0.05>
                    double diff = -.05 + .1 * random.NextDouble();
                    foreach (var l in layers)
                        foreach (var n in l.Neurons)
                            n.FixWeight(diff);
                    deltaDiffCount = 0;
                }
                else
                    deltaDiffCount++;
            }
        }

        void FeedForward()
        {
            for (int i = 0; i < layers[0].Neurons.Length; i++)
                layers[0].Neurons[i].Output = inputs[i];

            for (int i = 1; i < layers.Length; i++)
                for (int j = 0; j < layers[i].Neurons.Length; j++)
                {
                    layers[i].Neurons[j].Input = layers[i - 1].Neurons.Select(neuron => neuron.Output * neuron.Weights[j].Value).Sum() + layers[i].Neurons[j].Biases;
                    layers[i].Neurons[j].Output = TanH(layers[i].Neurons[j].Input);
                }
        }

        public bool FeedForward(double[] Inputs, params double[] expected)
        {
            for (int i = 0; i < layers[0].Neurons.Length; i++)
                layers[0].Neurons[i].Output = Inputs[i];

            for (int i = 1; i < layers.Length; i++)
                for (int j = 0; j < layers[i].Neurons.Length; j++)
                {
                    layers[i].Neurons[j].Input = layers[i - 1].Neurons.Select(neuron => neuron.Output * neuron.Weights[j].Value).ToArray().Sum() + layers[i].Neurons[j].Biases;
                    layers[i].Neurons[j].Output = TanH(layers[i].Neurons[j].Input);
                }

            //ShowOutput(expected);

            return delta < 0.2;
        }

        double Sigmoid(double value)
        {
            return 1.0 / (1.0 + Math.Pow(Math.E, -value));
        }

        double TanH(double value)
        {
            return Math.Tanh(value);
        }

        public void ShowNetwork()
        {
            int i = 0;
            for (i = 0; i < layers.Length-1; i++)
            {
                Console.WriteLine($"{i + 1} layer");
                foreach(var neuron in layers[i].Neurons)
                    Console.WriteLine($"Neuron: {neuron.Output}, Biases: {neuron.Biases}, Weights: {String.Join(", ", neuron.Weights.Select(w => w.Value).ToArray())}");
            }
            Console.WriteLine($"{i + 1} layer");
            foreach (var neuron in layers[i].Neurons)
                Console.WriteLine($"Neuron: {neuron.Output}, Biases: {neuron.Biases}");
            Console.WriteLine($"Delta = {delta}\n");
        }

        public void ShowOutput(params double[] expected)
        {
            if (expected.Length != 0)
            {
                delta = layers[layers.Length - 1].Neurons.Zip(expected, (n, e) => Math.Pow(n.Output - e, 2)).Sum();
                for (int i = 0; i < layers.Last().Neurons.Length; i++)
                    Console.WriteLine($"Output: {layers.Last().Neurons[i].Output} Expected: {expected[i]}");
                Console.WriteLine($"Delta: {delta}\n");
            }
            else
                foreach (var neuron in layers.Last().Neurons)
                    Console.WriteLine($"Output: {neuron.Output} Delta: {delta}");
        }

        public void ExportWeights(string path)
        {
            int layersCount = layers.Length;
            string weights = "";
            for (int i = 0; i < layersCount - 1; i++)
            {
                weights += String.Join(',', layers[i].Neurons.Select(n=>String.Join(',', n.Weights.Select(w => w.Value).ToArray())).ToArray()) + '\n';
            }
            File.WriteAllText(path, weights);
        }
        public void ExportBiases(string path)
        {
            int layersCount = layers.Length;
            string biases = "";
            for (int i = 0; i < layersCount; i++)
                biases += String.Join(',', layers[i].Neurons.Select(n => n.Biases).ToArray()) + '\n';
            File.WriteAllText(path, biases);
        }

        public void ImportWeights(string path)
        {
            try
            {
                string[] weights = File.ReadAllLines(path);
                for (int i = 0; i < weights.Length - 1; i++)
                    layers[i].InitilizeCustomWeights(weights[i].Split(',').Select(w => double.Parse(w)).ToArray());
            }
            catch
            {
                throw new Exception("Can't set custom weights. File format is invalid. ");
            }

        }
        public void ImportBiases(string path)
        {
            try
            {
                string[] biases = File.ReadAllLines(path);
                double[][] CustomBiases = biases.Select(line => line.Split(',').Select(b => double.Parse(b)).ToArray()).ToArray();
                for (int i = 0; i < layers.Length; i++)
                    for (int j = 0; j < layers[i].Neurons.Length; j++)
                        layers[i].Neurons[j].Biases = CustomBiases[i][j];
            }
            catch
            {
                throw new Exception("Can't set custom biases. File format is invalid. ");
            }
        }

        class Layer
        {
            int neuronCount;
            int outputCount;
            double eta;

            public Neuron[] Neurons { get; set; }

            public double[] Gamma { get; set; }
            public double[] Error { get; set; }

            public Layer(int inputsCount, int outputCount, Random random)
            {
                this.eta = 0.05;
                this.neuronCount = inputsCount;
                this.outputCount = outputCount;
                Neurons = new Neuron[inputsCount];
                for (int i = 0; i < inputsCount; i++)
                {
                    Neurons[i] = new Neuron(random);
                    Neurons[i].Weights = new Synapse[outputCount];
                    Neurons[i].DeltaWeights = new Synapse[outputCount];
                    for (int j = 0; j < outputCount; j++)
                    {
                        Neurons[i].Weights[j] = new Synapse();
                        Neurons[i].DeltaWeights[j] = new Synapse();
                    }
                }

                Gamma = new double[inputsCount];
                Error = new double[inputsCount];

                InitilizeWeights(random);
            }
            
            public void InitilizeWeights(Random random)
            {
                //Rozkład Bottou dla sigmoid a = 2.83
                double min = -(2.83 / Math.Sqrt(outputCount));
                double max = (2.83 / Math.Sqrt(outputCount));
                foreach (var neuron in Neurons)
                    foreach (var w in neuron.Weights)
                        w.Value = min + (max - min) * (double)random.NextDouble() - 0.5f;
            }

            public void InitilizeCustomWeights(double[] custom)
            {
                int customCount = -1;
                for (int i = 0; i < Neurons.Length; i++)
                    for (int j = 0; j < Neurons[i].Weights.Length; j++)
                        Neurons[i].Weights[j].Value = custom[++customCount];
            }
            double DerivativeSigmoid(double value)
            {
                //version for neuron's output
                return value * (1.0 - value);
            }
            double DerivativeTanH(double value)
            {
                //version for neuron's input
                return 1.0 / Math.Pow(Math.Cosh(value), 2);
            }

            public void BackPropOutput(double[] expected, ref Layer previousLayer)
            {
                for (int i = 0; i < neuronCount; i++)
                {
                    Error[i] = 2*(Neurons[i].Output - expected[i]);
                    Gamma[i] = Error[i] * DerivativeTanH(Neurons[i].Input);
                }

                //Caluclating detla weights
                for (int j = 0; j < previousLayer.neuronCount; j++)
                    for (int i = 0; i < neuronCount; i++)
                    {
                       previousLayer.Neurons[j].Biases -= eta * Gamma[i];
                       previousLayer.Neurons[j].DeltaWeights[i].Value = eta * Gamma[i] * previousLayer.Neurons[j].Output;
                       previousLayer.Neurons[j].Weights[i].Value -= previousLayer.Neurons[j].DeltaWeights[i].Value;
                    }
            }
            public void BackPropHidden(ref Layer previousLayer, Layer nextLayer)
            {
                //Caluclate new gamma using gamma sums of the forward layer
                for (int i = 0; i < neuronCount; i++)
                {
                    //tutaj error to suma gamm z następnej warstwy wagi 
                    for (int j = 0; j < nextLayer.Gamma.Length; j++)
                            Error[i] += nextLayer.Gamma[j] * Neurons[i].Weights[j].Value;
                    Gamma[i] = Error[i] * DerivativeTanH(Neurons[i].Input);
                    //Neurons[i].Output * (1 - Neurons[i].Output) to pochodna z sigmoid po wartości wejściowej do neuronu
                }

                //Caluclating detla weights
                for (int j = 0; j < previousLayer.neuronCount; j++)
                    for (int i = 0; i < neuronCount; i++)
                    {
                        previousLayer.Neurons[j].Biases -= eta * Gamma[i];
                        previousLayer.Neurons[j].DeltaWeights[i].Value = eta * Gamma[i] * previousLayer.Neurons[j].Output;
                        previousLayer.Neurons[j].Weights[i].Value -= previousLayer.Neurons[j].DeltaWeights[i].Value;
                    }
            }
        }

        class Neuron
        {
            public double Output { get; set; }
            public double Input { get; set; }
            public double Biases { get; set; }

            public Synapse[] Weights { get; set; }
            
            public Synapse[] DeltaWeights { get; set; }

            public Neuron(Random random)
            {
                Biases = (double)random.NextDouble() - 0.5f;
            }

            public void FixWeight()
            {
                for (int i = 0; i < Weights.Length; i++)
                {
                    Weights[i].Value -= DeltaWeights[i].Value;
                    Biases -= DeltaWeights[i].Value / Output;
                }
            }
            public void FixWeight(double d)
            {
                for (int i = 0; i < Weights.Length; i++)
                {
                    Weights[i].Value -= d;
                }
            }
        }

        class Synapse
        {
            public double Value { get; set; }
        }
    }
}
