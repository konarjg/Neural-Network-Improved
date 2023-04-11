using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

public class Matrix
{
    public int Rows { protected set; get; }
    public int Columns { protected set; get; }
    
    public double[,] Values { protected set; get; }

    public Matrix(int rows, int columns)
    {
        Values = new double[rows, columns];
        Rows = rows;
        Columns = columns;
    }

    public Matrix(double[,] values)
    {
        Values = values;
        Rows = values.GetLength(0);
        Columns = values.GetLength(1);
    }

    public void FillRandomly()
    {
        for (int x = 0; x < Rows; ++x)
        {
            for (int y = 0; y < Columns; ++y)
                Values[x, y] = 0.1f * x + x * Math.Pow(-1, y);
        }
    }

    public double this[int row, int column]
    {
        get { return Values[row, column]; } 
        set { Values[row, column] = value; }
    }

    public double[] GetRow(int row)
    {
        var r = new double[Columns];

        for (int y = 0; y < Columns; ++y)
            r[y] = Values[row, y];

        return r;
    }

    public double[] GetColumn(int column)
    {
        var c = new double[Rows];

        for (int x = 0; x < Rows; ++x)
            c[x] = Values[x, column];

        return c;
    }

    public override string ToString()
    {
        var s = "";

        for (int x = 0; x < Rows; ++x)
        {
            s += "|";

            for (int y = 0; y < Columns; ++y)
                s += Values[x, y] + " ";

            s = s.Remove(s.Length - 1);
            s += "|\n";
        }

        return s.Remove(s.Length - 1);
    }

    public static Matrix Parse(string m)
    {
        m = m.Replace("|", "");

        var reader = new StringReader(m);
        var line = "";
        var elements = new List<double>();

        int rows = 0;
        int columns = 0;

        while ((line = reader.ReadLine()) != null)
        {
            var row = line.Split(' ');
            columns = row.Length;
            ++rows;

            for (int i = 0; i < row.Length; ++i)
                elements.Add(double.Parse(row[i]));
        }

        var A = new Matrix(rows, columns);

        for (int x = 0; x < rows; ++x)
        {
            for (int y = 0; y < columns; ++y)
                A[x, y] = elements[x * columns + y];
        }

        return A;
    }

    #region Operations

    private static double Multiply(double[] row, double[] column)
    {
        var a = 0d;

        for (int i = 0; i < row.Length; ++i)
            a += row[i] * column[i];

        return a;
    }

    public static Matrix operator !(Matrix A)
    {
        var B = new Matrix(A.Columns, A.Rows);

        for (int x = 0; x < A.Rows; ++x)
        {
            for (int y = 0; y < A.Columns; ++y)
                B[y, x] = A[x, y];
        }

        return B;
    }

    public static Matrix operator +(Matrix A, Matrix B)
    {
        if (A.Rows != B.Rows || A.Columns != B.Columns)
            throw new ArgumentOutOfRangeException();

        var C = new Matrix(A.Rows, B.Columns);

        for (int x = 0; x < A.Rows; ++x)
        {
            for (int y = 0; y < B.Columns; ++y)
                C[x, y] = A[x, y] + B[x, y];
        }

        return C;
    }

    public static Matrix operator -(Matrix A)
    {
        var C = new Matrix(A.Rows, A.Columns);

        for (int x = 0; x < A.Rows; ++x)
        {
            for (int y = 0; y < A.Columns; ++y)
                C[x, y] = -A[x, y];
        }

        return C;
    }

    public static Matrix operator -(Matrix A, Matrix B)
    {
        return A + -B;
    }

    public static Matrix operator *(Matrix A, double b)
    {
        var C = new Matrix(A.Rows, A.Columns);

        for (int x = 0; x < A.Rows; ++x)
        {
            for (int y = 0; y < A.Columns; ++y)
                C[x, y] = A[x, y] * b;
        }

        return C;
    }

    public static Matrix operator /(Matrix A, double b)
    {
        var C = new Matrix(A.Rows, A.Columns);

        for (int x = 0; x < A.Rows; ++x)
        {
            for (int y = 0; y < A.Columns; ++y)
                C[x, y] = A[x, y] / b;
        }

        return C;
    }

    public static Matrix operator *(Matrix A, Matrix B)
    {
        var C = new Matrix(A.Rows, B.Columns);

        for (int y = 0; y < B.Columns; ++y)
        {
            for (int x = 0; x < A.Rows; ++x)
                C[x, y] = Multiply(A.GetRow(x), B.GetColumn(y));
        }

        return C;
    }

    //Hadamard product
    public static Matrix operator %(Matrix A, Matrix B)
    {
        if (A.Rows != B.Rows || A.Columns != B.Columns)
            throw new ArgumentException();

        var C = new Matrix(A.Rows, A.Columns);

        for (int y = 0; y < A.Columns; ++y)
        {
            for (int x = 0; x < A.Rows; ++x)
                C[x, y] = A[x, y] * B[x, y];
        }

        return C;
    }

    //Kronecker product
    public static Matrix operator ^(Matrix A, Matrix B)
    {
        var C = new Matrix(A.Rows * B.Rows, A.Columns * B.Columns);

        for (int x = 0; x < A.Rows; ++x)
        {
            for (int y = 0; y < B.Rows; ++y)
            {
                for (int i = 0; i < A.Columns; ++i)
                {
                    for (int j = 0; j < B.Columns; ++j)
                        C[x * B.Rows + y, i * B.Columns + j] = A[x, i] * B[y, j];
                }
            }
        }

        return C;
    }

    #endregion
}
