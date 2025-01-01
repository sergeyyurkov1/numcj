package numcj

@Test
func N2darray_shape() {
    let a = np.n2darray([[1.0, 1.0], [1.0, 1.0]])

    @Expect(a.shape[0], 2)
    @Expect(a.shape[1], 2)
}

@Test
func N2darray_size() {
    let a = np.n2darray([[1.0, 1.0], [1.0, 1.0]])

    @Expect(a.size, 4)
}

@Test
class N2darray_transpose_tests {
    let a = np.n2darray([[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]])

    @TestCase
    func T_test() {
        @Expect(a.T.toString(), "[[0.200000, 0.500000, -0.260000], [0.800000, -0.910000, -0.270000], [-0.500000, 0.260000, 0.170000], [1.000000, -0.500000, 0.870000]]")
    }

    @TestCase
    func transpose_test() {
        @Expect(a.transpose().toString(), "[[0.200000, 0.500000, -0.260000], [0.800000, -0.910000, -0.270000], [-0.500000, 0.260000, 0.170000], [1.000000, -0.500000, 0.870000]]")
    }
}

@Test
class N2darray_operator_tests_n2darrays {
    let a = np.n2darray([[1.0, 1.0], [1.0, 1.0]])
    let b = np.n2darray([[2.0, 2.0], [2.0, 2.0]])

    @TestCase
    func addition_test() {
        let c = a + b
        @Expect(c.toString(), "[[3.000000, 3.000000], [3.000000, 3.000000]]")
    }

    @TestCase
    func subtraction_test() {
        let c = a - b

        @Expect(c.toString(), "[[-1.000000, -1.000000], [-1.000000, -1.000000]]")
    }

    @TestCase
    func multiplication_test() {
        let c = a * b

        @Expect(c.toString(), "[[2.000000, 2.000000], [2.000000, 2.000000]]")
    }

    @TestCase
    func division_test() {
        let c = a / b

        @Expect(c.toString(), "[[0.500000, 0.500000], [0.500000, 0.500000]]")
    }
}

@Test
class N2darray_operator_tests_n2darrays_and_1darrays {
    let a = np.n2darray([[1.0, 1.0], [1.0, 1.0]])
    let b = Vector1([2.0, 2.0])

    @TestCase
    func addition_test() {
        let c = a + b
        @Expect(c.toString(), "[[3.000000, 3.000000], [3.000000, 3.000000]]")
    }

    @TestCase
    func subtraction_test() {
        let c = a - b

        @Expect(c.toString(), "[[-1.000000, -1.000000], [-1.000000, -1.000000]]")
    }
}

@Test
class N2darray_operator_tests_n2darrays_and_floats {
    let a = np.n2darray([[1.0, 1.0], [1.0, 1.0]])
    let b = 3.0

    @TestCase
    func multiplication_test() {
        let c = a * b

        @Expect(c.toString(), "[[3.000000, 3.000000], [3.000000, 3.000000]]")
    }

    @TestCase
    func division_test() {
        let c = a / b

        @Expect(c.toString(), "[[0.333333, 0.333333], [0.333333, 0.333333]]")
    }
}
