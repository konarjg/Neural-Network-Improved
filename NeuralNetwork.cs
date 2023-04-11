using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

public enum Activation
{
    LINEAR,
    SIGMOID,
    TANH,
    RELU,
    SOFTMAX
}

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

public class NeuralNetwork
{
    public Layer InputLayer;
    public Layer[] HiddenLayers;
    public Layer OutputLayer;
    public Activation Activation;
    public Activation OutputActivation;

    public NeuralNetwork(Activation activation, Activation outputActivation, int inputNeurons, int outputNeurons, params int[] hiddenNeurons)
    {
        Activation = activation;
        OutputActivation = outputActivation;

        InputLayer = new Layer(inputNeurons, null);

        if (hiddenNeurons.Length == 0)
            OutputLayer = new Layer(outputNeurons, InputLayer);
        else
        {
            HiddenLayers = new Layer[hiddenNeurons.Length];

            HiddenLayers[0] = new Layer(hiddenNeurons[0], InputLayer);

            for (int i = 1; i < hiddenNeurons.Length; ++i)
                HiddenLayers[i] = new Layer(hiddenNeurons[i], HiddenLayers[i - 1]);

            OutputLayer = new Layer(outputNeurons, HiddenLayers[hiddenNeurons.Length - 1]);
        }
    }

    private double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    private Matrix Sigmoid(Matrix u)
    {
        var A = new Matrix(u.Rows, u.Columns);

        for (int x = 0; x < u.Rows; ++x)
        {
            for (int y = 0; y < u.Columns; ++y)
                A[x, y] = Sigmoid(u[x, y]);
        }

        return A;
    }

    private Matrix dSigmoid(Matrix u)
    {
        var A = new Matrix(u.Rows, u.Columns);

        for (int x = 0; x < u.Rows; ++x)
        {
            for (int y = 0; y < u.Columns; ++y)
                A[x, y] = Sigmoid(u[x, y]) * (1 - Sigmoid(u[x, y]));
        }

        return A;
    }

    private double Relu(double x)
    {
        return Math.Max(x, 0);
    }

    private Matrix Relu(Matrix u)
    {
        var A = new Matrix(u.Rows, u.Columns);

        for (int x = 0; x < u.Rows; ++x)
        {
            for (int y = 0; y < u.Columns; ++y)
                A[x, y] = Relu(u[x, y]);
        }

        return A;
    }

    private Matrix dRelu(Matrix u)
    {
        var A = new Matrix(u.Rows, u.Columns);

        for (int x = 0; x < u.Rows; ++x)
        {
            for (int y = 0; y < u.Columns; ++y)
                A[x, y] = u[x, y] > 0 ? 1 : 0;
        }

        return A;
    }

    private Matrix Softmax(Matrix u)
    {
        var A = new Matrix(u.Rows, u.Columns);
        var sum = 0d;

        for (int i = 0; i < u.Rows; ++i)
            sum += Math.Exp(u[i, 0]);

        for (int i = 0; i < u.Rows; ++i)
            A[i, 0] = Math.Exp(u[i, 0]) / sum;

        return A;
    }

    private Matrix dSoftmax(Matrix u)
    {
        var A = new Matrix(u.Rows, u.Columns);
        var S = Softmax(u);
        
        for (int i = 0; i < u.Rows; ++i)
        {
            for (int j = 0; j < u.Rows; ++j)
            {
                if (i == j)
                    A[j, 0] = S[i, 0] * (1 - S[j, 0]);
                else
                    A[j, 0] = -S[j, 0] * S[i, 0];
            }
        }

        return A;
    }

    private double Tanh(double x)
    {
        return (Math.Exp(x) - Math.Exp(-x)) / (Math.Exp(x) + Math.Exp(-x));
    }

    private Matrix Tanh(Matrix u)
    {
        var A = new Matrix(u.Rows, u.Columns);

        for (int x = 0; x < u.Rows; ++x)
        {
            for (int y = 0; y < u.Columns; ++y)
                A[x, y] = Tanh(u[x, y]);
        }

        return A;
    }

    private Matrix dTanh(Matrix u)
    {
        var A = new Matrix(u.Rows, u.Columns);

        for (int x = 0; x < u.Rows; ++x)
        {
            for (int y = 0; y < u.Columns; ++y)
                A[x, y] = 1d - Math.Pow(Tanh(u[x, y]), 2);
        }

        return A;
    }

    public Matrix F(Matrix u)
    {
        switch (Activation)
        {
            case Activation.LINEAR:
                return u;

            case Activation.SIGMOID:
                return Sigmoid(u);

            case Activation.TANH: 
                return Tanh(u);

            case Activation.RELU:
                return Relu(u);

            case Activation.SOFTMAX: 
                return Softmax(u);
        }

        throw new NotImplementedException();
    }

    public Matrix dF(Matrix u)
    {
        switch (Activation)
        {
            case Activation.LINEAR:
                return new Matrix(new double[,] { { 1 } });

            case Activation.SIGMOID:
                return dSigmoid(u);

            case Activation.TANH:
                return dTanh(u);

            case Activation.RELU:
                return dRelu(u);

            case Activation.SOFTMAX:
                return dSoftmax(u);
        }

        throw new NotImplementedException();
    }

    public Matrix G(Matrix u)
    {
        switch (OutputActivation)
        {
            case Activation.LINEAR:
                return u;

            case Activation.SIGMOID:
                return Sigmoid(u);

            case Activation.TANH:
                return Tanh(u);

            case Activation.RELU:
                return Relu(u);

            case Activation.SOFTMAX:
                return Softmax(u);
        }

        throw new NotImplementedException();
    }

    public Matrix dG(Matrix u)
    {
        switch (OutputActivation)
        {
            case Activation.LINEAR:
                return new Matrix(new double[,] { { 1 } });

            case Activation.SIGMOID:
                return dSigmoid(u);

            case Activation.TANH:
                return dTanh(u);

            case Activation.RELU:
                return dRelu(u);

            case Activation.SOFTMAX:
                return dSoftmax(u);
        }

        throw new NotImplementedException();
    }

    public Matrix Calculate(Matrix input)
    {
        if (input.Rows != InputLayer.Neurons)
            throw new ArgumentOutOfRangeException();

        var y = F(HiddenLayers[0].Weights * input + HiddenLayers[0].Biases);

        for (int i = 1; i < HiddenLayers.Length; ++i)
            y = F(HiddenLayers[i].Weights * y + HiddenLayers[i].Biases);

        y = G(OutputLayer.Weights * y + OutputLayer.Biases);

        return y;
    }

    public Matrix[] CalculateActivations(Matrix input)
    {
        if (input.Rows != InputLayer.Neurons)
            throw new ArgumentOutOfRangeException();

        var result = new List<Matrix>();
        result.Add(input);

        var y = F(HiddenLayers[0].Weights * input + HiddenLayers[0].Biases);
        result.Add(y);

        for (int i = 1; i < HiddenLayers.Length; ++i)
        {
            y = F(HiddenLayers[i].Weights * y + HiddenLayers[i].Biases);
            result.Add(y);
        }

        y = G(OutputLayer.Weights * y + OutputLayer.Biases);
        result.Add(y);

        return result.ToArray();
    }

    public Task Train(Matrix[] inputs, Matrix[] expectedOutputs, double learningRate, double momentum, int epochs, string trainingOutputFilePath)
    {
        return Task.Factory.StartNew(() =>
        {
            if (inputs.Length != expectedOutputs.Length)
                throw new ArgumentException();

            var text = "";

            for (int epoch = 0; epoch < epochs; ++epoch)
            {
                var W3 = OutputLayer.Weights;

                Matrix dW3 = null;
                Matrix dB3 = null;
                Matrix[] dW2 = new Matrix[HiddenLayers.Length];
                Matrix[] dB2 = new Matrix[HiddenLayers.Length];

                for (int i = 0; i < inputs.Length; ++i)
                {
                    var input = inputs[i];
                    var output = Calculate(input);
                    var activations = CalculateActivations(input);
                    var epsilon = output - expectedOutputs[i];

                    var delta3 = epsilon % dG(OutputLayer.Weights * activations[activations.Length - 2] + OutputLayer.Biases);
                    var delta2 = new Matrix[HiddenLayers.Length];

                    int k = activations.Length - 3;
                    delta2[HiddenLayers.Length - 1] = (!OutputLayer.Weights * delta3) % dF(HiddenLayers[HiddenLayers.Length - 1].Weights * activations[k] + HiddenLayers[HiddenLayers.Length - 1].Biases);

                    --k;

                    for (int j = HiddenLayers.Length - 2; j >= 0; --j)
                    {
                        delta2[j] = (!HiddenLayers[j + 1].Weights * delta2[j + 1]) % dF(HiddenLayers[j].Weights * activations[k] + HiddenLayers[j].Biases);

                        --k;
                    }

                    if (i == 0)
                    {
                        dW3 = !activations[activations.Length - 2] ^ delta3;
                        dB3 = delta3;
                        k = activations.Length - 3;

                        for (int j = HiddenLayers.Length - 1; j >= 0; --j)
                        {
                            dW2[j] = !activations[k] ^ delta2[j];
                            dB2[j] = delta2[j];
                            --k;
                        }
                    }
                    else
                    {
                        dW3 += !activations[activations.Length - 2] ^ delta3;
                        dB3 += delta3;
                        k = activations.Length - 3;

                        for (int j = HiddenLayers.Length - 1; j >= 0; --j)
                        {
                            dW2[j] += !activations[k] ^ delta2[j];
                            dB2[j] += delta2[j];
                            --k;
                        }
                    }
                }

                for (int j = HiddenLayers.Length - 1; j >= 0; --j)
                {
                    HiddenLayers[j].AdjustWeights(dW2[j] / inputs.Length, (learningRate + momentum));
                    HiddenLayers[j].AdjustBiases(dB2[j] / inputs.Length, (learningRate + momentum));
                }

                OutputLayer.AdjustWeights(dW3 / inputs.Length, (learningRate + momentum));
                OutputLayer.AdjustBiases(dB3 / inputs.Length, (learningRate + momentum));
            }
        });
    }

    public void SaveState(string path)
    {
        var text = "";

        for (int i = 0; i < HiddenLayers.Length; ++i)
            text += HiddenLayers[i].Weights + "\n!\n";

        text += OutputLayer.Weights + "\n!\n";

        for (int i = 0; i < HiddenLayers.Length; ++i)
            text += HiddenLayers[i].Biases + "\n!\n";

        text += OutputLayer.Biases + "\n!\n";

        File.WriteAllText(path, text);
    }

    public void LoadState(string path)
    {
         var lines = File.ReadAllLines(path);
        var matrices = new List<string>();

        var m = "";

        for (int i = 0; i < lines.Length; ++i)
        {
            if (lines[i].Contains("!"))
            {
                matrices.Add(m);
                m = "";
                continue;
            }

            m += lines[i] + "\n";
        }

        int j = 0;

        for (int i = 0; i < HiddenLayers.Length; ++i)
        {
            HiddenLayers[i].Weights = Matrix.Parse(matrices[j]);
            ++j;
        }

        OutputLayer.Weights = Matrix.Parse(matrices[j]);
        ++j;

        for (int i = 0; i < HiddenLayers.Length; ++i)
        {
            HiddenLayers[i].Biases = Matrix.Parse(matrices[j]);
            ++j;
        }

        OutputLayer.Biases = Matrix.Parse(matrices[j]);
    }
}
