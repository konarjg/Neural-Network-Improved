using System.Net.Http.Headers;
using static System.Net.Mime.MediaTypeNames;

static class Program
{
    public static Random Random = new Random();

    private static double Encode(char character)
    {
        if (character >= '0' && character <= '9')
            return character - '0';

        if (character == '^')
            return 10;

        return 11;
    }

    private static double[] Encode(string equation)
    {
        var code = new double[equation.Length];

        for (int i = 0; i < equation.Length; ++i)
            code[i] = Encode(equation[i]);

        return code;
    }

    private static async void Train()
    {
        var network = new Network(2, 2, 3);

        if (File.Exists("state.txt"))
            network.LoadState("state.txt");

        var data = new Dictionary<Matrix, Matrix>();

        for (double j = -20; j <= 20; ++j)
        {
            if (j == 0)
                continue;

            for (double i = 0; i <= 20; ++i)
            {
                var input = new Matrix(new double[,] { { j }, { i } });
                var output = new Matrix(new double[,] { { j }, { i + 1 } });

                data.Add(input, output);
            }
        }

        await network.Train(data.Keys.ToArray(), data.Values.ToArray(), 0.001d, 0d, 1000, "training.txt");

        var text = "";

        for (int x = -10; x <= 10; ++x)
        {
            if (x == 0)
                continue;

            for (int y = 0; y <= 10; ++y)
            {
                var z = network.Calculate(new Matrix(new double[,] { { x }, { y } }));
                text += string.Format("|{0}x^{1} dx = {2}*1/{3}x^{3} + C\n", x, y, z[0, 0], z[1, 0]);
            }
        }

        File.WriteAllText("output.txt", text);
        network.SaveState("state.txt");
    }

    public static void Main(string[] args)
    {
        /*Train();
        while (Console.ReadKey().Key != ConsoleKey.Escape) { }
        */

        var network = new Network(2, 2, 3);
        network.LoadState("state.txt");

        for (double x = -10; x <= 10; ++x)
        {
            if (x == 0)
                continue;

            for (double y = 0; y <= 10; ++y)
            {
                var z = network.Calculate(new Matrix(new double[,] { { x }, { y } }));
                var z1 = Math.Round(z[0, 0]) / Math.Round(z[1, 0]);
                var z2 = Math.Round(z[1, 0]);

                var xs = (y == 0 ? x.ToString() : x == 1 ? "x" : x.ToString() + "x");
                var ys = (y == 1 || y == 0 ? "" : "^" + y.ToString());
                var z1s = (z2 == 0 ? "1" : z1 == 1 ? "x" : z1.ToString() + "x");
                var z2s = (z2 == 1 || z2 == 0 ? "" : "^" + z2.ToString());

                Console.WriteLine("|{0}{1} dx = {2}{3}", xs, ys, z1s, z2s);
            }
        }
    }
}