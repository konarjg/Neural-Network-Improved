using System.Net.Http.Headers;

static class Program
{
    public static Random Random = new Random();

    public static void Main(string[] args)
    {
        var network = new Network(2, 1, 3);

        var data = new Dictionary<Matrix, double>
        {
            { new Matrix(new double[,] { { 1 }, { 1 } }), 2 },
            { new Matrix(new double[,] { { 2 }, { 1 } }), 3 },
            { new Matrix(new double[,] { { 2 }, { 2 } }), 4 },
            { new Matrix(new double[,] { { 3 }, { 2 } }), 5 },
            { new Matrix(new double[,] { { 3 }, { 3 } }), 6 },
            { new Matrix(new double[,] { { 4 }, { 3 } }), 7 },
            { new Matrix(new double[,] { { 4 }, { 4 } }), 8 },
            { new Matrix(new double[,] { { 5 }, { 4 } }), 9 },
            { new Matrix(new double[,] { { 5 }, { 5 } }), 10 },
            { new Matrix(new double[,] { { 6 }, { 5 } }), 11 },
        };

        network.Train(data.Keys.ToArray(), data.Values.ToArray(), 0.025d, 0d, 500);

        Console.WriteLine(network.Calculate(new Matrix(new double[,] { { 10 }, { 5 } })));
    }
}