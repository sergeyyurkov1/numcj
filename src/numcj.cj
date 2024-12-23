package numcj

import matrix4cj.*
import std.collection.*
import std.math.*

/** Custom type aliases.
 */
type Vector3 = Array<Array<Array<Float64>>> // 3D array

type Vector2 = Array<Array<Float64>> // 2D array

type Vector1 = Array<Float64> // 1D array

class Numcj {
    public func zeros(shape: (Int64, Int64)) {
        /** Initializes an N2Darray with zeros.
         * @param shape Controls the shape of the N2Darray: (2, 2) -> [[0.0, 0.0], [0.0, 0.0]]
         * @return New N2Darray
         * @exception "" ""
         */
        return N2Darray(shape)
    }

    public func randn(shape: (Int64, Int64)) {
        /** Initializes an N2Darray with random values.
         * @param shape Controls the shape of the N2Darray: (2, 2) -> [[,], [,]]
         * @return New N2Darray
         * @exception "" ""
         */
        return N2Darray(shape, random: true)
    }

    public func dot(a: N2Darray, b: N2Darray) {
        /** Dot product of two N2Darrays.
         * @param a Left N2Darray
         * @param b Right N2Darray
         * @return New N2Darray
         * @exception "" ""
         */
        return N2Darray(a.M.times(b.M))
    }

    public func broadcastTo(array: N2Darray, shape: Int64): N2Darray {
        /** Broadcast an array to a new shape.
         * @param array N2Darray to broadcast
         * @param shape The number of columns to broadcast to
         * @return New N2Darray
         * @exception "" ""
         */
        let a = Matrix(Vector2(shape, {_ => array.array[0]}))
        return N2Darray(a)
    }

    // Factories to create N2Darrays
    public func n2darray(shape: (Int64, Int64)) {
        return N2Darray(shape)
    }

    public func n2darray(shape: (Int64, Int64), v: Float64) {
        return N2Darray(shape, v: v)
    }

    public func n2darray(m: Matrix) {
        return N2Darray(m)
    }

    public func n2darray(a: Vector2) {
        return N2Darray(a)
    }

    // Factory to create a N3Darray
    public func n3darray(shape: (Int64, Int64, Int64)) {
        return N3Darray(shape)
    }

    public func maximum(x: Float64, array: N2Darray) {
        let output = ArrayList<Array<Float64>>([])
        for (a in array.array) {
            output.append(a |> map {i => max(x, i)} |> collectArray)
        }
        return N2Darray(output.toArray())
    }
    public func maximum(x: Float64, array: Vector1) {
        return N2Darray(array |> map {i => max(x, i)} |> collectArray)
    }
    public func exp(array: N2Darray) {
        let output = ArrayList<Array<Float64>>([])
        for (a in array.array) {
            output.append(a |> map {i => exp(i)} |> collectArray)
        }
        return N2Darray(output.toArray())
    }
    public func sum(array: N2Darray, axis!: Int64 = 0, keepdims!: Bool = false) {
        match (axis) {
            case 0 =>
                var a = Matrix(Vector2(1, {_ => Vector1(array.array[0].size, {_ => 0.0})}))
                for (i in array.array) {
                    a.plusEquals(Matrix(i))
                }
                return N2Darray(a)
            case 1 => if (keepdims) {
                var a = ArrayList<Vector1>([])
                for (i in array.array) {
                    let s = i |> reduce(this.sum_)
                    a.append([s.getOrThrow()])
                }
                return N2Darray(a.toArray())
            } else {
                var a = ArrayList<Float64>([])
                for (i in array.array) {
                    let s = i |> reduce(this.sum_)
                    a.append(s.getOrThrow())
                }
                return N2Darray(a.toArray())
            }
            case _ => throw Exception("'axis' can only be 0 or 1.")
        }
    }
    private func sum_(a: Float64, b: Float64) {
        return a + b
    }
}
