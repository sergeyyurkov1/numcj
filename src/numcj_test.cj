package numcj

import std.unittest.*
import matrix4cj.*

let np = Numcj(seed: 0)

@Test
func Numcj_n2darray_shape() {
    let a = np.n2darray((2, 2))

    @Expect(a.toString(), "[[0.000000, 0.000000], [0.000000, 0.000000]]")
}

@Test
func Numcj_n2darray_shape_value() {
    let a = np.n2darray((2, 2), 1.0)

    @Expect(a.toString(), "[[1.000000, 1.000000], [1.000000, 1.000000]]")
}

@Test
func Numcj_n2darray_array() {
    let a = np.n2darray([[1.0, 1.0], [1.0, 1.0]])

    @Expect(a.toString(), "[[1.000000, 1.000000], [1.000000, 1.000000]]")
}

@Test
func Numcj_n2darray_matrix() {
    let a = np.n2darray(Matrix([[1.0, 1.0], [1.0, 1.0]]))

    @Expect(a.toString(), "[[1.000000, 1.000000], [1.000000, 1.000000]]")
}

@Test
func Numcj_n2darray_empty_initialization() {
    @ExpectThrows[Exception]({
        let a = np.n2darray()
    })
}

@Test
func Numcj_zeros() {
    let a = np.zeros((2, 2))

    @Expect(a.toString(), "[[0.000000, 0.000000], [0.000000, 0.000000]]")
}

@Test
func Numcj_randn() {
    let a = np.randn((2, 2))

    println(a.toString())

    @Expect(a.toString(), "[[0.256808, 0.913389], [0.037365, 0.669069]]")
}

@Test
func Numcj_dot() {
    let a = np.n2darray([[1.0, 2.0, 3.0, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]])

    let b = np.n2darray([[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]])

    let c = np.dot(a, b.T)

    @Expect(c.toString(), "[[2.800000, -1.790000, 1.885000], [6.900000, -4.810000, -0.300000], [-0.590000, -1.949000, -0.474000]]")
}

@Test
func Numcj_broadcastTo() {
    let a = np.n2darray([2.000000, 3.000000, 0.500000])

    let b = np.broadcastTo(a, 3)

    @Expect(b.toString(), "[[2.000000, 3.000000, 0.500000], [2.000000, 3.000000, 0.500000], [2.000000, 3.000000, 0.500000]]")
}

@Test
func Numcj_maximum_array() {
    let a = Array<Float64>([0.0, 2.0, -1.0, 3.3, -2.7, 1.1, 2.2, -100.0])

    let b = np.maximum(0.0, np.n2darray(a))

    @Expect(b.toString(), "[[0.000000, 2.000000, 0.000000, 3.300000, 0.000000, 1.100000, 2.200000, 0.000000]]")
}

@Test
func Numcj_maximum_n2darray() {
    let a = np.n2darray([0.0, 2.0, -1.0, 3.3, -2.7, 1.1, 2.2, -100.0])

    let b = np.maximum(0.0, a)

    @Expect(b.toString(), "[[0.000000, 2.000000, 0.000000, 3.300000, 0.000000, 1.100000, 2.200000, 0.000000]]")
}

@Test
func Numcj_exp() {
    let a = np.n2darray([[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]])

    let b = np.exp(a)

    @Expect(b.toString(), "[[121.510418, 3.353485, 10.859063], [7331.973539, 0.163654, 1.221403], [4.095955, 2.860510, 1.026341]]")
}

@Test
func Numcj_sum_axis0() {
    let a = np.n2darray([[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]])

    let b = np.sum(a, axis: 0)

    @Expect(b.toString(), "[[15.110000, 0.451000, 2.611000]]")
}

@Test
func Numcj_sum_axis1() {
    let a = np.n2darray([[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]])

    let b = np.sum(a, axis: 1)

    @Expect(b.toString(), "[[8.395000, 7.290000, 2.487000]]")
}

@Test
func Numcj_sum_axis1_keepdims() {
    let a = np.n2darray([[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]])

    let b = np.sum(a, axis: 1, keepdims: true)

    @Expect(b.toString(), "[[8.395000], [7.290000], [2.487000]]")
}

@Test[i in [ 
((2.0, 3.0, 5, true), "[2.000000, 2.250000, 2.500000, 2.750000, 3.000000]"), 
((0.0, 3.0, 5, true), "[0.000000, 0.750000, 1.500000, 2.250000, 3.000000]"), 
((2.0, 3.0, 5, false), "[2.000000, 2.200000, 2.400000, 2.600000, 2.800000]"), 
((0.0, 3.0, 5, false), "[0.000000, 0.600000, 1.200000, 1.800000, 2.400000]") 
 ] 
]
func Numcj_linspace(i: ((Float64, Float64, Int64, Bool), String)) {
    let a = np.linspace(i[0][0], i[0][1], i[0][2], endpoint: i[0][3])

    print(a.toString())

    @Expect(a.toString() == i[1])
}

// @Test
// class AddTests {
//     @TestCase
//     func addTest() {
//         @Expect(add(2, 3), 5)
//     }

//     @TestCase
//     func addZero() {
//         @Expect(add(2, 0), 2)
//     }
// }
