using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

public class Layer
{
    public int Neurons { private set; get; }
    public Matrix Weights;
    public Matrix Biases;

    public Layer(int neurons, Layer previous)
    {
        Neurons = neurons;

        if (previous != null)
        {
            Weights = new Matrix(neurons, previous.Neurons);
            Biases = new Matrix(neurons, 1);

            Weights.FillRandomly();
            Biases.FillRandomly();
        }
        else
        {
            Weights = new Matrix(0, 0);
            Biases = new Matrix(0, 0);
        }
    }

    public void AdjustWeights(Matrix gradient, double learningRate) 
    { 
        for (int x = 0; x < Weights.Rows; ++x)
        {
            for (int y = 0; y < Weights.Columns; ++y)
                Weights[x, y] -= learningRate * gradient[x, y];
        }
    }

    public void AdjustBiases(Matrix gradient, double learningRate)
    {
        for (int x = 0; x < Biases.Rows; ++x)
        {
            for (int y = 0; y < Biases.Columns; ++y)
                Biases[x, y] -= learningRate * gradient[x, y];
        }
    }
}

public class Network
{
    public Layer InputLayer;
    public Layer HiddenLayer;
    public Layer OutputLayer;

    public Network(int inputNeurons, int outputNeurons, int hiddenNeurons)
    {
        InputLayer = new Layer(inputNeurons, null);
        HiddenLayer = new Layer(hiddenNeurons, InputLayer);
        OutputLayer = new Layer(outputNeurons, HiddenLayer);
    }

    private double Elu(double x)
    {
        if (x < 0)
            return Math.Exp(x) - 1;

        return x;
    }

    private Matrix Elu(Matrix u)
    {
        var A = new Matrix(u.Rows, u.Columns);

        for (int x = 0; x < u.Rows; ++x)
        {
            for (int y = 0; y < u.Columns; ++y)
                A[x, y] = Elu(u[x, y]);
        }

        return A;
    }

    private Matrix dElu(Matrix u)
    {
        var A = new Matrix(u.Rows, u.Columns);

        for (int x = 0; x < u.Rows; ++x)
        {
            for (int y = 0; y < u.Columns; ++y)
                A[x, y] = (A[x, y] > 0 ? 1 : Math.Exp(A[x, y]));
        }

        return A;
    }

    public Matrix Calculate(Matrix input)
    {
        if (input.Rows != InputLayer.Neurons)
            throw new ArgumentOutOfRangeException();

        var y1 = Elu(HiddenLayer.Weights * input + HiddenLayer.Biases);
        var y = OutputLayer.Weights * y1 + OutputLayer.Biases;

        return y;
    }

    public void Train(Matrix[] inputs, double[] expectedOutputs, double learningRate, double momentum, int epochs)
    {
        if (inputs.Length != expectedOutputs.Length)
            throw new ArgumentException();

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            var W3 = OutputLayer.Weights;
            var W2 = HiddenLayer.Weights;
            var B2 = HiddenLayer.Biases;

            Matrix dW3 = null;
            Matrix dB3 = null;
            Matrix dW2 = null;
            Matrix dB2 = null;

            Console.WriteLine("-------------------");
            Console.WriteLine("Epoch {0}: ", epoch + 1);

            for (int i = 0; i < inputs.Length; ++i)
            {
                var input = inputs[i];
                var output = Calculate(input);
                var epsilon = output[0, 0] - expectedOutputs[i];
               
                Console.WriteLine("{0} + {1} = {2}, Błąd: {3}%", input[0, 0], input[1, 0], output[0, 0], Math.Abs(Math.Round((output[0, 0] - expectedOutputs[i]) / output[0, 0], 3)));

                var y1 = Elu(W2 * input + B2);
                var delta3 = epsilon;
                var delta2 = (!W3 * delta3) % dElu(W2 * input + B2);

                if (i == 0)
                {
                    dW3 = !y1 * delta3;
                    dW2 = !input ^ delta2;
                    dB3 = new Matrix(new double[,] { { delta3 } });
                    dB2 = delta2;
                }
                else
                {
                    dW3 += !y1 * delta3;
                    dW2 += !input ^ delta2;
                    dB3 += new Matrix(new double[,] { { delta3 } });
                    dB2 += delta2;
                }
            }

            HiddenLayer.AdjustWeights(dW2 / inputs.Length, (learningRate + momentum));
            OutputLayer.AdjustWeights(dW3 / inputs.Length, (learningRate + momentum));
            HiddenLayer.AdjustBiases(dB2 / inputs.Length, (learningRate + momentum));
            OutputLayer.AdjustBiases(dB3 / inputs.Length, (learningRate + momentum));
        }

        Console.WriteLine("");
    }
}
