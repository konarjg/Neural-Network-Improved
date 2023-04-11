using System.Net.Http.Headers;
using System.Runtime.Serialization.Formatters;
using static System.Net.Mime.MediaTypeNames;
using static System.Runtime.InteropServices.JavaScript.JSType;

static class Program
{
    public static Random Random = new Random();

    private static List<string> Functions = new List<string>()
    {
        "exp",
        "ln",
        "sin",
        "cos",
        "tg",
        "ctg",
        "atan"
    };

    private static Dictionary<string, int> Operators = new Dictionary<string, int>()
    {
        { "^", 3 },
        { "*", 2 },
        { "/", 2 },
        { "+", 1 },
        { "-", 1 },
    };


    private static string AddSpaces(string equation)
    {
        var result = "";

        for (int i = 0; i < equation.Length; ++i)
        {
            if (Operators.Keys.ToList().Contains(equation[i].ToString()) || equation[i] == '(' || equation[i] == ')')
                result += " " + equation[i] + " ";
            else
                result += equation[i];
        }

        return result;
    }

    private static Queue<string> ToRPN(string equation)
    {
        equation = AddSpaces(equation);
        var result = new Queue<string>();
        var stack = new Stack<string>();
        var tokens = equation.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var number = 0d;
        var o2 = "";

        for (int i = 0; i < tokens.Length; ++i)
        {
            switch (tokens[i])
            {
                case string s when double.TryParse(s, out number) || s.Contains("x"):
                    result.Enqueue(s);
                    break;

                case string s when Functions.Contains(s):
                    stack.Push(s);
                    break;

                case string s when Operators.ContainsKey(s):
                    while (stack.TryPeek(out o2) && Operators.ContainsKey(o2) && Operators[s] <= Operators[o2])
                        result.Enqueue(stack.Pop());

                    stack.Push(s);
                    break;

                case string s when s == "(":
                    stack.Push(s);
                    break;

                case string s when s == ")":
                    while (stack.TryPeek(out o2))
                    {
                        if (o2 == "(")
                        {
                            stack.Pop();

                            if (Functions.Contains(o2))
                            {
                                result.Enqueue(stack.Pop());
                                break;
                            }
                        }

                        if (stack.Count != 0)
                            result.Enqueue(stack.Pop());
                    }

                    break;
            }
        }

        return result;
    }

    private static double Encode(string token)
    {
        var n = 0d;

        if (double.TryParse(token, out n) || token.Contains("x"))
            return 1;

        double c = 1;

        for (int i = 0; i < Operators.Count; ++i)
        {
            if (Operators.ElementAt(i).Key.Contains(token.Replace(" ", "")))
                return c;

            ++c;
        }

        for (int i = 0; i < Functions.Count; ++i)
        {
            if (Functions.ElementAt(i).Contains(token.Replace(" ", "")))
                return c;

            ++c;
        }

        return c;
    }

    private static Matrix Encode(string equation, int length)
    {
        var code = new double[length];
        var rpn = ToRPN(equation);

        var i = 0;
        var s = "";

        while (rpn.TryDequeue(out s))
        {
            code[i] = Encode(s);
            ++i;
        }

        while (i < length)
        {
            code[i] = 0;
            ++i;
        }

        var result = new Matrix(length, 1);

        for (int k = 0; k < code.Length; ++k)
            result[k, 0] = code[k];

        return result;
    }

    private static string ToMethod(Matrix x)
    {
        switch (x[0, 0])
        {
            case double a when Math.Round(a) == 1:
                return "Całka podstawowa";

            case double a when Math.Round(a) == 2:
                return "Całka przez podstawianie";

            case double a when Math.Round(a) == 3:
                return "Całka przez części";

            case double a when Math.Round(a) == 4:
                return "Całka wymierna";

            case double a when Math.Round(a) == 5:
                return "Całka trygonometryczna";
        }

        return "Niezdefiniowana";
    }

    private static void PrepareTrainingSet()
    {
        var data = "";

        data += string.Format("2*x^2 1\n");
        data += string.Format("2*sin(x) 1\n");
        data += string.Format("2*cos(x) 1\n");
        data += string.Format("2*exp(x) 1\n");
        data += string.Format("2*sin(2*x) 2\n");
        data += string.Format("2*cos(2*x) 2\n");
        data += string.Format("2*exp(2*x) 2\n");
        data += string.Format("2*atan(x) 3\n");
        data += string.Format("2*ln(x) 3\n");
        data += string.Format("2*x^2*exp(x) 3\n");
        data += string.Format("2*x^2*sin(x) 3\n");
        data += string.Format("2*x^2*cos(x) 3\n");
        data += string.Format("2*exp(x)*cos(x) 3\n");
        data += string.Format("2*exp(x)*sin(x) 3\n");
        data += string.Format("2*x/2*x^2 4\n");
        data += string.Format("1/(x^2+1) 4\n");
        data += string.Format("1/(2*x^3+2*x+1) 4\n");
        data += string.Format("sin(x)^2 5\n");
        data += string.Format("cos(x)^2 5\n");
        data += string.Format("sin(x)*cos(x) 5\n");

        File.WriteAllText("data.txt", data);
    }

    private static async void Train(NeuralNetwork network)
    {
        PrepareTrainingSet();

        var dataDecoded = File.ReadAllLines("data.txt");

        if (File.Exists("state.txt"))
            network.LoadState("state.txt");

        var data = new Dictionary<Matrix, Matrix>();
        var errors = 0d;
        var accuracy = 1d;
        var prevAccuracy = 0d;

        for (int j = 0; j < dataDecoded.Length; ++j)
        {
            var input = Encode(dataDecoded[j].Split(' ')[0], 20);
            var output = new Matrix(new double[,] { { double.Parse(dataDecoded[j].Split(' ')[1]) } });

            data.Add(input, output);
        }

        Console.Write("Docelowa dokładność (%): ");
        var goal = double.Parse(Console.ReadLine());
        Console.Write("Współczynnik nauki: ");
        var rate = double.Parse(Console.ReadLine());

        do
        {
            accuracy = 1d;
            errors = 0d;

            for (int i = 0; i < dataDecoded.Length; ++i)
            {
                var input = data.ElementAt(i).Key;
                var output = data.ElementAt(i).Value;
                var prediction = network.Calculate(input);

                if (ToMethod(prediction) != ToMethod(output))
                    ++errors;
            }

            accuracy -= errors / dataDecoded.Length;

            if (accuracy != prevAccuracy)
            {
                Console.Clear();
                Console.WriteLine("Docelowa dokładność (%): {0}", goal);
                Console.WriteLine("Współczynnik nauki: {0}\n", rate);

                Console.WriteLine("Dokładność: {0}%", accuracy * 100);
            }

            if (accuracy * 100 >= goal)
                break;

            await network.Train(data.Keys.ToArray(), data.Values.ToArray(), rate, 0, 100, "training.txt");
            prevAccuracy = accuracy;
        }
        while (accuracy * 100 < goal);

        Console.WriteLine("Trening zakończony!\n");
        network.SaveState("state.txt");
    }

    private static string Display(Queue<string> rpn)
    {
        var s = "";
        var result = "";

        while (rpn.TryDequeue(out s))
            result += s + " ";

        return result.TrimEnd();
    }

    public static void Main(string[] args)
    {
        var network = new NeuralNetwork(Activation.SIGMOID, Activation.LINEAR, 20, 1, 10, 5);

        /*Train(network);

        while (Console.ReadKey().Key != ConsoleKey.Escape) { }*/

        if (File.Exists("state.txt"))
            network.LoadState("state.txt");

        var dataDecoded = File.ReadAllLines("data.txt");
        var errors = 0d;
        var accuracy = 1d;
        var mean = 0d;
        var variance = 0d;

        for (int i = 0; i < dataDecoded.Length; ++i)
        {
            var input = dataDecoded[i].Split(' ')[0];
            var output = new Matrix(new double[,] { { double.Parse(dataDecoded[i].Split(' ')[1]) } });
            var prediction = network.Calculate(Encode(input, 20));

            if (Math.Round(prediction[0, 0]) != output[0, 0])
                ++errors;

            mean += (prediction[0, 0] - output[0, 0]) / prediction[0, 0];
        }

        accuracy -= errors / dataDecoded.Length;
        mean /= dataDecoded.Length;

        for (int i = 0; i < dataDecoded.Length; ++i)
        {
            var input = Encode(dataDecoded[i].Split(' ')[0], 20);
            var output = new Matrix(new double[,] { { double.Parse(dataDecoded[i].Split(' ')[1]) } });
            var prediction = network.Calculate(input);

            variance += Math.Pow((prediction[0, 0] - output[0, 0]) - mean, 2);
        }

        variance /= dataDecoded.Length - 1;

        Console.WriteLine("Dokładność: {0}%", accuracy * 100);
        Console.WriteLine("Średni błąd: {0}%", Math.Abs(mean) * 100);
        Console.WriteLine("Wariancja: {0}", variance);
        Console.WriteLine("Odchylenie standardowe: {0}\n", Math.Sqrt(variance));

        do
        {
            Console.Write("Podaj całkę: ");

            var integral = Console.ReadLine();
            Console.WriteLine("|{0} dx -> {1}", integral, ToMethod(network.Calculate(Encode(integral, 20))));
        }
        while (Console.ReadKey().Key != ConsoleKey.Escape);
    }
}