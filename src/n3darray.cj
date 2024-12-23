package numcj

public class N3Darray {
    var array: Vector3

    public init(shape: (Int64, Int64, Int64)) {
        this.array = Vector3(shape[0], {_ => Vector2(shape[1], {_ => Vector1(shape[2], {_ => 0.0})})})
    }

    public func toString() {
        return this.array
    }
}
