using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Serialization;
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

public class RegressionModel
{
    public Layer InputLayer;
    public Layer[] HiddenLayers;
    public Layer OutputLayer;

    public RegressionModel(int inputNeurons, int outputNeurons, params int[] hiddenNeurons)
    {
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
                A[x, y] = (u[x, y] > 0 ? 1 : Math.Exp(u[x, y]));
        }

        return A;
    }

    public Matrix Calculate(Matrix input)
    {
        if (input.Rows != InputLayer.Neurons)
            throw new ArgumentOutOfRangeException();

        var y = Elu(HiddenLayers[0].Weights * input + HiddenLayers[0].Biases);

        for (int i = 1; i < HiddenLayers.Length; ++i)
            y = Elu(HiddenLayers[i].Weights * y + HiddenLayers[i].Biases);

        y = OutputLayer.Weights * y + OutputLayer.Biases;

        return y;
    }

    public Matrix[] CalculateActivations(Matrix input)
    {
        if (input.Rows != InputLayer.Neurons)
            throw new ArgumentOutOfRangeException();

        var result = new List<Matrix>();
        result.Add(input);

        var y = Elu(HiddenLayers[0].Weights * input + HiddenLayers[0].Biases);

        result.Add(y);

        for (int i = 1; i < HiddenLayers.Length; ++i)
        {
            y = Elu(HiddenLayers[i].Weights * y + HiddenLayers[i].Biases);
            result.Add(y);
        }

        y = OutputLayer.Weights * y + OutputLayer.Biases;
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

                var error = new Matrix(OutputLayer.Neurons, 1);

                for (int i = 0; i < error.Rows; ++i)
                    error[i, 0] = 0;

                for (int i = 0; i < inputs.Length; ++i)
                    error += Calculate(inputs[i]) - expectedOutputs[i];

                error = error / (double)inputs.Length;

                text += string.Format("\nEpoch {0}: {1}\n", epoch + 1, error);
                Console.WriteLine("Epoch {0}: {1}", epoch + 1, error);
              
                for (int i = 0; i < inputs.Length; ++i)
                {
                    var input = inputs[i];
                    var output = Calculate(input);
                    var activations = CalculateActivations(input);
                    var epsilon = output - expectedOutputs[i];

                    var delta3 = epsilon;
                    var delta2 = new Matrix[HiddenLayers.Length];

                    int k = activations.Length - 3;
                    delta2[HiddenLayers.Length - 1] = (!W3 * delta3) % dElu(HiddenLayers[HiddenLayers.Length - 1].Weights * activations[k] + HiddenLayers[HiddenLayers.Length - 1].Biases);
                  
                    --k;

                    for (int j = HiddenLayers.Length - 2; j >= 0; --j)
                    {
                        delta2[j] = (!HiddenLayers[j + 1].Weights * delta2[j + 1]) % dElu(HiddenLayers[j].Weights * activations[k] + HiddenLayers[j].Biases);

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

            Console.WriteLine("Training finished!");

            File.WriteAllText(trainingOutputFilePath, text);
        });
    }

    public void SaveState(string path)
    {
        var text = "";

        for (int i = 0; i < HiddenLayers.Length; ++i)
            text += HiddenLayers[i].Weights + "!\n";

        text += OutputLayer.Weights + "!\n";

        for (int i = 0; i < HiddenLayers.Length; ++i)
            text += HiddenLayers[i].Biases + "!\n";

        text += OutputLayer.Biases + "!\n";

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
