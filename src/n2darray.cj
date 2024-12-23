package numcj

import matrix4cj.*
import std.collection.*
import std.random.*

let r = Random()

public class N2Darray {
    var array: Vector2
    var M: Matrix

    public init(shape: (Int64, Int64), v!: Float64 = 0.0, random!: Bool = false) {
        /** Initializes an N2Darray, stores both the matrix and the array to switch between if the matrix doesn't support a particular operation.
         * @param shape Controls the shape of the N2Darray: (2, 2) -> [[0.0, 0.0], [0.0, 0.0]]
         * @param v Controls the default value to initialize N2Darray with
         * @param random Controls whether to initialize N2Darray with random values
         * @exception "" ""
         */
        if (random) {
            this.array = Vector2(shape[0], {_ => Vector1(shape[1], {_ => r.nextFloat64()})})
        } else {
            this.array = Vector2(shape[0], {_ => Vector1(shape[1], {_ => v})})
        }
        this.M = Matrix(this.array) // Matrix can only be Array<Array<Float64>>
    }
    public init(a: Vector2) {
        /** Initializes an N2Darray, stores both the matrix and the array to switch between if the matrix doesn't support a particular operation.
         * @param a Initializes an N2Darray from a 2D array
         * @exception "" ""
         */
        this.array = a
        this.M = Matrix(a)
    }
    public init(m: Matrix) {
        /** Initializes an N2Darray, stores both the matrix and the array to switch between if the matrix doesn't support a particular operation.
         * @param m Initializes an N2Darray from a Matrix
         * @exception "" ""
         */
        this.M = m
        this.array = m.getArray()
    }

    // PROPERTIES
    prop shape: (Int64, Int64) {
        /** array.shape; works as a property
         * @return A tuple of row and column dimensions of the internal matrix
         * @exception "" ""
         */
        get() {
            return (this.M.getRowDimension(), this.M.getColumnDimension())
        }
    }
    prop size: Int64 {
        /** array.size; works as a property
         * @return The number of total values in the internal matrix
         * @exception "" ""
         */
        get() {
            return this.M.getRowDimension() * this.M.getColumnDimension()
        }
    }
    prop T: N2Darray {
        /** array.T Transposes the internal matrix; works as a property
         * @return A new N2Darray with the transposed internal matrix
         * @exception "" ""
         */
        get() {
            return N2Darray(this.M.transpose())
        }
    }

    // METHODS
    public func toString() {
        /** Printable representation of the N2Darray
         * @return Internal arrays
         */
        return this.M.getArray()
    }
    public func transpose(): N2Darray {
        /** array.transpose() Transposes the internal matrix
         * @return A new N2Darray with the transposed internal matrix
         * @exception "" ""
         */
        return N2Darray(this.M.transpose())
    }

    // OPERATORS
    public operator func +(that: N2Darray): N2Darray {
        /** Array-wise addition of two N2Darrays
         * @return New N2Darray
         */
        if (that.array.size == 1) {
            let a = Numcj().broadcastTo(that, this.array.size)
            return N2Darray(this.M.plus(a.M))
        }
        return N2Darray(this.M.plus(that.M))
    }
    public operator func +(that: Array<Float64>): N2Darray {
        /** Array-wise addition of an N2Darray and an Array with conversion
         * @return New N2Darray
         */
        let a = Matrix(Vector2(this.array.size, {_ => that}))
        return N2Darray(this.M.plus(a))
    }
    public operator func -(that: N2Darray): N2Darray {
        /** Array-wise subtraction of two N2Darrays
         * @return New N2Darray
         */
        return N2Darray(this.M.minus(that.M))
    }
    public operator func *(that: N2Darray): N2Darray {
        /** Array-wise multiplication of two N2Darrays
         * @return New N2Darray
         */
        return N2Darray(this.M.arrayTimes(that.M))
    }
    public operator func *(v: Float64): N2Darray {
        /** Multiplication of an N2Darray and a Float64
         * @return New N2Darray
         */
        let a = this.M.getArrayCopy()

        for (i in 0..a.size) {
            for (j in 0..a[i].size) {
                a[i][j] *= v
            }
        }

        return N2Darray(Matrix(a))
    }
    public operator func /(that: N2Darray): N2Darray {
        /** Array-wise division of two N2Darrays controlling for shape mismatch; when array B's row size == 1, performs element-wise division
         * @return New N2Darray
         * @exception Exception Throws an exception if array A's row size doesn't match array B's row size when `array A's row size > array B's row size > 1`
         */
        if (this.array[0].size == that.array[0].size) {
            return N2Darray(this.M.arrayRightDivide(that.M))
        } else if (that.array[0].size == 1) {
            let output = ArrayList<Vector1>([])
            for ((i, j) in this.array |> zip(that.array)) {
                output.append(i |> map {k => k / j[0]} |> collectArray)
            }
            return N2Darray(output.toArray())
        } else {
            throw Exception(
                "Cannot divide N2Darray of size ${this.array[0].size} by N2Darray of size ${that.array[0].size}.")
        }
    }
    public operator func /(v: Float64): N2Darray {
        /** Division of an N2Darray and a Float64
         * @return New N2Darray
         */
        let a = this.M.getArrayCopy()

        for (i in 0..a.size) {
            for (j in 0..a[i].size) {
                a[i][j] /= v
            }
        }

        return N2Darray(Matrix(a))
    }
}
