package com.gengoai.apollo.linear.v2;

import com.gengoai.Copyable;
import com.gengoai.Validation;
import com.gengoai.collection.Iterators;
import com.gengoai.collection.Streams;
import com.gengoai.json.JsonEntry;
import com.gengoai.json.JsonSerializable;
import com.gengoai.math.Math2;
import com.gengoai.math.Operator;
import com.gengoai.math.Optimum;
import com.gengoai.tuple.Tuple2;
import org.apache.commons.math3.util.FastMath;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import java.io.Serializable;
import java.util.*;
import java.util.function.*;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.apollo.linear.v2.NDArrayFactory.DENSE;
import static com.gengoai.collection.Iterators.zipWithIndex;
import static com.gengoai.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public abstract class NDArray implements Copyable<NDArray>, Serializable, JsonSerializable, Iterable<NDArray.Entry> {
   private static final long serialVersionUID = 1L;
   protected int[] shape;
   private int order;
   private long length;
   private int matrixLength;
   private int numSlices;


   public NDArray(int[] shape) {
      this.shape = new int[]{1, 1, 1, 1};
      System.arraycopy(shape, 0, this.shape, 0, shape.length);
      this.order = Arrays.stream(shape).map(i -> i > 1 ? 1 : 0).sum();
      this.matrixLength = this.shape[0] * this.shape[1];
      this.numSlices = this.shape[2] * this.shape[3];
      this.length = this.matrixLength * this.numSlices;
   }

   public static NDArray fromJson(JsonEntry entry) {
      if (entry.getBooleanProperty("dense")) {
         return DenseNDArray.fromJson(entry);
      }
      NDArray ndArray = DENSE.zeros(entry.getValProperty("shape")
                                         .asIntegerValueArray());
      JsonEntry array = entry.getProperty("data");
      zipWithIndex(array.elementIterator()).forEachRemaining(e -> {
         NDArray matrix = ndArray.slice(e.getValue());
         zipWithIndex(e.getKey().elementIterator())
            .forEachRemaining(v -> matrix.set(v.getValue(), v.getKey().getAsFloat()));
      });
      return ndArray;
   }

   public NDArray T() {
      return sliceOperation(v -> {
         NDArray out = getFactory().zeros(v.columns(), v.rows());
         v.forEach(e -> out.set(e.getColumn(), e.getRow(), e.getValue()));
         return out;
      });
   }

   public NDArray add(float scalar) {
      return mapScalar(getFactory().zeros(shape()), scalar, Operator::add);
   }

   public NDArray add(NDArray other) {
      return map(newZeroArray(),
                 other,
                 Operator::add);
   }

   public NDArray add(NDArray other, Axis axis) {
      return mapVector(getFactory().zeros(other.shape()),
                       other,
                       axis,
                       Operator::add);
   }

   public NDArray addi(float scalar) {
      return mapScalar(this, scalar, Operator::add);
   }

   public NDArray addi(NDArray other) {
      return mapVector(other, Operator::add);
   }

   public NDArray addi(NDArray other, Axis axis) {
      return mapVector(this, other, axis, Operator::add);
   }

   public int[] argMax(Axis axis) {
      return argOptimum(axis, Optimum.MAXIMUM);
   }

   public int[] argMin(Axis axis) {
      return argOptimum(axis, Optimum.MINIMUM);
   }

   private int[] argOptimum(Axis axis, Optimum optimum) {
      checkArgument(axis.isRowOrColumn(), "Axis (" + axis + ") not supported");
      checkArgument(order <= 2, "Order (" + order + ") not supported");
      int[] out = new int[dimension(axis)];
      double[] optimums = new double[dimension(axis)];
      Arrays.fill(optimums, optimum.startingValue());
      forEach(e -> {
         if (optimum.test(e.getValue(), optimums[e.getIndex(axis)])) {
            optimums[e.getIndex(axis)] = e.getValue();
            out[e.getIndex(axis)] = e.getIndex(axis.T());
         }
      });
      return out;
   }

   public int channels() {
      return dimension(Axis.CHANNEL);
   }

   public int columns() {
      return dimension(Axis.COLUMN);
   }

   public NDArray compress() {
      return this;
   }

   /**
    * Calculates the variance-covariance matrix of this NDArray
    *
    * @return The variance-covariance matrix
    */
   public NDArray cov() {
      return sliceOperation(v -> {
         NDArray c = v.sub(getFactory().ones(v.rows(), v.rows())
                                       .mmul(v)
                                       .muli(1f / v.rows()));
         return c.T().mmul(c).divi(v.rows());
      });
   }

   public NDArray decrement(int[] indices, float value) {
      return set(indices, get(indices) - value);
   }

   public NDArray decrement(int row, int column, int kernel, int channel, float value) {
      return set(row, column, kernel, channel, get(row, column, kernel, channel) - value);
   }

   public NDArray decrement(int row, int column, int kernel, float value) {
      return set(row, column, kernel, get(row, column, kernel) - value);
   }

   public NDArray decrement(int row, int column, float value) {
      return set(row, column, get(row, column) - value);
   }

   public NDArray decrement(int row, float value) {
      return set(row, get(row) - value);
   }

   public NDArray diag() {
      return sliceOperation(v -> {
         if (v.isScalar()) {
            return v.copy();
         }
         if (v.isVector()) {
            Axis axis = v.isColumnVector() ? Axis.ROW : Axis.COLUMN;
            NDArray out = getFactory().zeros(v.dimension(axis), v.dimension(axis));
            for (int i = 0; i < v.dimension(axis); i++) {
               out.set(i, i, v.get(i));
            }
            return out;
         }
         if (v.isSquare()) {
            NDArray out = getFactory().zeros(v.rows(), v.columns());
            for (int i = 0; i < v.rows(); i++) {
               if (i < v.columns()) {
                  out.set(i, i, v.get(i, i));
               }
            }
            return out;
         }
         throw new IllegalStateException("Rectangular slices are not supported");
      });
   }

   public int dimension(Axis axis) {
      return shape[axis.ordinal];
   }

   public NDArray div(NDArray other) {
      return map(newZeroArray(),
                 other,
                 Operator::divide);
   }

   public NDArray div(float value) {
      return mapScalar(newZeroArray(),
                       value,
                       Operator::divide);
   }

   public NDArray div(NDArray other, Axis axis) {
      return mapVector(newZeroArray(),
                       other,
                       axis,
                       Operator::divide);
   }

   public NDArray divi(float value) {
      return mapScalar(this, value, Operator::divide);
   }

   public NDArray divi(NDArray other) {
      return mapVector(other, Operator::divide);
   }

   public NDArray divi(NDArray other, Axis axis) {
      return mapVector(this,
                       other,
                       axis,
                       Operator::divide);
   }

   public NDArray fill(float value) {
      sliceStream().forEach(tuple -> tuple.v2.iterator().forEachRemaining(e -> e.setValue(value)));
      return this;
   }

   public void forEachSlice(Consumer<NDArray> sliceConsumer) {
      sliceStream().forEach(t -> sliceConsumer.accept(t.v2));
   }

   public void forEachSparse(Consumer<Entry> consumer) {
      sparseIterator().forEachRemaining(consumer);
   }

   public abstract float get(int... indices);

   public abstract NDArrayFactory getFactory();

   public NDArray getVector(int index, Axis axis) {
      int[] newShape = shape();
      newShape[axis.ordinal] = 1;
      NDArray out = getFactory().zeros(newShape);
      sliceStream().forEach(t -> {
         NDArray slice = out.slice(t.v1);
         for (int i = 0; i < dimension(axis.T()); i++) {
            if (axis == Axis.ROW) {
               slice.set(i, t.v2.get(index, i));
            } else {
               slice.set(i, t.v2.get(i, index));
            }
         }
      });
      return out;
   }

   public NDArray increment(int[] indices, float value) {
      return set(indices, get(indices) + value);
   }

   public NDArray increment(int row, int column, int kernel, int channel, float value) {
      return set(row, column, kernel, channel, get(row, column, kernel, channel) + value);
   }

   public NDArray increment(int row, int column, int kernel, float value) {
      return set(row, column, kernel, get(row, column, kernel) + value);
   }

   public NDArray increment(int row, int column, float value) {
      return set(row, column, get(row, column) + value);
   }

   public NDArray increment(int row, float value) {
      return set(row, get(row) + value);
   }

   public boolean isColumnVector() {
      return isMatrix() && dimension(Axis.ROW) > 1 && dimension(Axis.COLUMN) == 1;
   }

   public boolean isDense() {
      return false;
   }

   public boolean isMatrix() {
      return dimension(Axis.KERNEL) == 1 && dimension(Axis.CHANNEL) == 1;
   }

   public boolean isRowVector() {
      return isMatrix() && dimension(Axis.ROW) == 1 && dimension(Axis.COLUMN) > 1;
   }

   public boolean isScalar() {
      return dimension(Axis.ROW) == 1 && dimension(Axis.COLUMN) == 1 &&
                dimension(Axis.KERNEL) == 1 && dimension(Axis.CHANNEL) == 1;
   }

   public boolean isSparse() {
      return false;
   }

   public boolean isSquare() {
      return isMatrix() && rows() == columns();
   }

   public boolean isVector() {
      return isRowVector() || isColumnVector();
   }

   public Iterator<Entry> iterator() {
      return Iterators.transform(new IndicesIterator(), Entry::new);
   }

   public int kernels() {
      return dimension(Axis.KERNEL);
   }

   public long length() {
      return length;
   }

   public NDArray map(DoubleUnaryOperator operator) {
      return mapOperator(newZeroArray(), operator);
   }

   protected NDArray map(NDArray out, NDArray other, DoubleBinaryOperator operator) {
      if (other.isScalar()) {
         return mapScalar(out, other.get(0), operator);
      }
      if (other.isVector()) {
         return mapVector(out,
                          other,
                          other.isRowVector() ? Axis.COLUMN : Axis.ROW,
                          operator);
      }
      if (other.isMatrix()) {
         return mapMatrix(out, other, operator);
      }
      return mapTensor(out, other, operator);
   }

   public NDArray map(NDArray other, DoubleBinaryOperator operator) {
      return map(getFactory().zeros(shape()), other, operator);
   }

   protected NDArray mapMatrix(NDArray out, NDArray matrix, DoubleBinaryOperator operator) {
      sliceStream().forEach(t -> {
         NDArray s1 = t.v2;
         NDArray s2 = out.slice(t.v1);
         for (int j = 0; j < s1.sliceLength(); j++) {
            s2.set(j, (float) operator.applyAsDouble(s1.get(j), matrix.get(j)));
         }
      });
      return out;
   }

   protected NDArray mapOperator(NDArray out, DoubleUnaryOperator operator) {
      sliceStream().forEach(t -> {
         NDArray slice = out.slice(t.v1);
         t.v2.forEach(e -> slice.set(e.matrixIndex(), (float) operator.applyAsDouble(e.getValue())));
      });
      return out;
   }

   protected NDArray mapScalar(NDArray out, float scalar, DoubleBinaryOperator operator) {
      sliceStream().forEach(t -> {
         NDArray s1 = t.v2;
         NDArray s2 = out.slice(t.v1);
         for (int j = 0; j < s1.sliceLength(); j++) {
            s2.set(j, (float) operator.applyAsDouble(s1.get(j), scalar));
         }
      });
      return out;
   }

   public <T> List<T> mapSlices(Function<NDArray, ? extends T> function) {
      List<T> out = new ArrayList<>(numSlices);
      IntStream.range(0, numSlices).forEach(i -> out.add(null));
      sliceStream().forEach(t -> out.set(t.v1, function.apply(t.v2)));
      return out;
   }

   protected NDArray mapTensor(NDArray out, NDArray tensor, DoubleBinaryOperator operator) {
      checkArgument(tensor.sliceLength() == sliceLength(),
                    "Length of each slice is not the same. (" + sliceLength() + ") != (" + tensor.sliceLength() + ")");
      checkArgument(slices() == tensor.slices(),
                    "Number of slices does not match. (" + slices() + ") != (" + tensor.slices() + ")");
      sliceStream().forEach(t -> {
         NDArray s1 = t.v2;
         NDArray s2 = tensor.slice(t.v1);
         NDArray sOut = out.slice(t.v1);
         for (int j = 0; j < s1.sliceLength(); j++) {
            sOut.set(j, (float) operator.applyAsDouble(s1.get(j), s2.get(j)));
         }
      });
      return out;
   }

   public NDArray mapVector(NDArray other, DoubleBinaryOperator operator) {
      return map(this, other, operator);
   }

   public NDArray mapVector(NDArray other, Axis axis, DoubleBinaryOperator operator) {
      return mapVector(this,
                       other,
                       axis,
                       operator);
   }

   protected NDArray mapVector(NDArray out, NDArray rowVector,
                               Axis axis, DoubleBinaryOperator operator
                              ) {
      sliceStream().forEach(t -> {
         NDArray s1 = t.v2;
         NDArray s2 = out.slice(t.v1);
         for (int row = 0; row < s1.dimension(Axis.ROW); row++) {
            for (int column = 0; column < s1.dimension(Axis.COLUMN); column++) {
               s2.set(row, column, (float) operator.applyAsDouble(s1.get(row, column), rowVector.get(
                  axis.T().select(row, column)
                                                                                                    )));
            }
         }
      });
      return out;
   }

   public NDArray mapi(DoubleUnaryOperator operator) {
      return mapOperator(this, operator);
   }

   public float max() {
      return optimum(Optimum.MAXIMUM);
   }

   public NDArray max(Axis axis) {
      return optimum(axis, Optimum.MAXIMUM);
   }

   public float mean() {
      return sum() / length;
   }

   public NDArray mean(Axis axis) {
      return sum(axis).divi(dimension(axis.T()));
   }

   public float min() {
      return optimum(Optimum.MINIMUM);
   }

   public NDArray min(Axis axis) {
      return optimum(axis, Optimum.MINIMUM);
   }

   public abstract NDArray mmul(NDArray other);

   public NDArray mul(NDArray other) {
      return copy().muli(other);
   }

   public NDArray mul(float value) {
      return mapScalar(getFactory().zeros(shape()), value, Operator::multiply);
   }

   public NDArray muli(float value) {
      return mapScalar(this, value, Operator::multiply);
   }

   public NDArray muli(NDArray other) {
      return mapVector(other, Operator::multiply);
   }

   private NDArray newZeroArray() {
      return getFactory().zeros(shape);
   }

   public float norm1() {
      return (float) sliceStream().mapToDouble(t ->
                                                  Streams.asStream(t.v2)
                                                         .mapToDouble(e -> Math.abs(e.getValue()))
                                                         .sum()
                                              ).sum();
   }

   public float norm2() {
      return (float) Math.sqrt(sumOfSquares());
   }

   protected NDArray optimum(Axis axis, Optimum optimum) {
      Validation.checkArgument(axis == Axis.ROW || axis == Axis.COLUMN,
                               "Only ROW and Axis.COLUMN supported");
      int[] newShape = shape();
      newShape[axis.T().ordinal] = 1;
      NDArray out = getFactory().constant((float) optimum.startingValue(), newShape);

      sliceStream().forEach(t -> {
         NDArray outSlice = out.slice(t.v1);
         t.v2.iterator().forEachRemaining(e -> {
            int i = axis.select(e.row, e.column);
            if (optimum.test(e.getValue(), outSlice.get(i))) {
               outSlice.set(i, e.getValue());
            }
         });
      });

      return out;
   }

   protected float optimum(Optimum optimum) {
      DoubleStream values = sliceStream().mapToDouble(t -> {
         double opt = optimum.startingValue();
         Iterator<Entry> iterator = t.v2.iterator();
         while (iterator.hasNext()) {
            float v = iterator.next().getValue();
            if (optimum.test(v, opt)) {
               opt = v;
            }
         }
         return opt;
      });

      if (optimum == Optimum.MAXIMUM) {
         return (float) values.max().orElse(Double.NaN);
      }

      return (float) values.min().orElse(Double.NaN);
   }

   public int order() {
      return order;
   }

   public NDArray pivot() {
      if (isSquare()) {
         NDArray p = getFactory().eye(rows());
         for (int i = 0; i < rows(); i++) {
            double max = get(i, i);
            int row = i;
            for (int j = i; j < rows(); j++) {
               if (get(j, i) > max) {
                  max = get(j, i);
                  row = j;
               }
            }

            if (i != row) {
               NDArray v = p.getVector(i, Axis.ROW);
               p.setVector(i, Axis.ROW, p.getVector(row, Axis.ROW));
               p.setVector(row, Axis.ROW, v);
            }
         }
         return p;
      } else if (order > 2) {
         NDArray[] out = new NDArray[numSlices];
         sliceStream().forEach(t -> out[t.v1] = t.v2.pivot());
         return getFactory().fromLayers(kernels(), channels(), out);
      }

      throw new IllegalArgumentException("Only square slices supported");
   }

   public NDArray rdiv(NDArray other) {
      return map(newZeroArray(),
                 other,
                 (v1, v2) -> v2 / v1);
   }

   public NDArray rdiv(float value) {
      return mapScalar(newZeroArray(),
                       value,
                       (v1, v2) -> v2 / v1);
   }

   public NDArray rdiv(NDArray other, Axis axis) {
      return mapVector(newZeroArray(),
                       other,
                       axis,
                       (v1, v2) -> v2 / v1);
   }

   public NDArray rdivi(NDArray other) {
      return map(this,
                 other,
                 (v1, v2) -> v2 / v1);
   }

   public NDArray rdivi(float value) {
      return mapScalar(this,
                       value,
                       (v1, v2) -> v2 / v1);
   }

   public NDArray rdivi(NDArray other, Axis axis) {
      return mapVector(this,
                       other,
                       axis,
                       (v1, v2) -> v2 / v1);
   }

   public int rows() {
      return dimension(Axis.ROW);
   }

   public NDArray rsub(NDArray other) {
      return map(this, other, (v1, v2) -> v2 - v1);
   }

   public NDArray rsub(float value) {
      return mapScalar(newZeroArray(), value, (v1, v2) -> v2 - v1);
   }

   public NDArray rsub(NDArray other, Axis axis) {
      return mapVector(newZeroArray(), other, axis, (v1, v2) -> v2 - v2);
   }

   public NDArray rsubi(float value) {
      return mapScalar(this, value, (v1, v2) -> v2 - v1);
   }

   public NDArray rsubi(NDArray other) {
      return map(this, other, (v1, v2) -> v2 - v1);
   }

   public NDArray rsubi(NDArray other, Axis axis) {
      return mapVector(this, other, axis, (v1, v2) -> v2 - v2);
   }

   public NDArray select(DoublePredicate predicate) {
      return mapOperator(newZeroArray(), v -> predicate.test(v) ? v : 0f);
   }

   public NDArray selecti(DoublePredicate predicate) {
      return mapOperator(this, v -> predicate.test(v) ? v : 0f);
   }

   public NDArray set(int[] indices, float value) {
      int[] dims = Util.ensureCorrectIndicies(indices);
      return set(dims[0], dims[1], dims[2], dims[3], value);
   }

   public abstract NDArray set(int row, int column, int kernel, int channel, float value);

   public NDArray set(int row, int column, int kernel, float value) {
      return set(row, column, kernel, 0, value);
   }

   public NDArray set(int row, int column, float value) {
      return set(row, column, 0, 0, value);
   }

   public NDArray set(int row, float value) {
      return set(row, 0, 0, 0, value);
   }

   protected abstract void setSlice(int slice, NDArray other);

   public NDArray setVector(int index, Axis axis, NDArray vector) {
      sliceStream().forEach(t -> {
         NDArray slice = vector.slice(t.v1);
         for (int i = 0; i < dimension(axis.T()); i++) {
            if (axis == Axis.ROW) {
               t.v2.set(index, i, slice.get(i));
            } else {
               t.v2.set(i, index, slice.get(i));
            }
         }
      });
      return this;
   }

   public int[] shape() {
      return Arrays.copyOf(shape, shape.length);
   }

   public NDArray slice(int kernel, int channel) {
      return slice(Util.index(kernel, dimension(Axis.KERNEL),
                              channel, dimension(Axis.CHANNEL)));
   }

   public abstract NDArray slice(int kernel);

   public int sliceLength() {
      return matrixLength;
   }

   public NDArray sliceMax() {
      return sliceOperation(v -> getFactory().zeros(1)
                                             .set(0, v.max()));
   }

   public NDArray sliceMean() {
      return sliceOperation(v -> getFactory().zeros(1)
                                             .set(0, v.mean()));
   }

   public NDArray sliceMin() {
      return sliceOperation(v -> getFactory().zeros(1)
                                             .set(0, v.min()));
   }

   public NDArray sliceNorm1() {
      return sliceOperation(v -> getFactory().zeros(1)
                                             .set(0, v.norm1()));
   }

   public NDArray sliceNorm2() {
      return sliceOperation(v -> getFactory().zeros(1)
                                             .set(0, v.norm2()));
   }

   public NDArray sliceOperation(Function<NDArray, NDArray> function) {
      NDArray[] out = new NDArray[numSlices];
      sliceStream().forEach(t -> out[t.v1] = function.apply(t.v2));
      return getFactory().fromLayers(kernels(), channels(), out);
   }

   public Stream<Tuple2<Integer, NDArray>> sliceStream() {
      return IntStream.range(0, slices()).mapToObj(i -> $(i, slice(i))).parallel();
   }

   public NDArray sliceSum() {
      return sliceOperation(v -> getFactory().zeros(1)
                                             .set(0, v.sum()));
   }

   public NDArray sliceSumOfSquares() {
      return sliceOperation(v -> getFactory().zeros(1)
                                             .set(0, v.sumOfSquares()));
   }

   public int slices() {
      return numSlices;
   }

   public Iterator<Entry> sparseIterator() {
      return Iterators.filter(iterator(), e -> e.getValue() != 0f);
   }

   public NDArray sub(NDArray other) {
      return map(newZeroArray(), other, Operator::subtract);
   }

   public NDArray sub(NDArray other, Axis axis) {
      return mapVector(newZeroArray(), other, axis, Operator::subtract);
   }

   public NDArray sub(float value) {
      return mapScalar(newZeroArray(), value, Operator::subtract);
   }

   public NDArray subi(float value) {
      return mapScalar(this, value, Operator::subtract);
   }

   public NDArray subi(NDArray other) {
      return map(this, other, Operator::subtract);
   }

   public NDArray subi(NDArray other, Axis axis) {
      return mapVector(this, other, axis, Operator::subtract);
   }

   public float sum() {
      return (float) sliceStream().mapToDouble(t -> {
         double sum = 0;
         Iterator<Entry> iterator = t.v2.sparseIterator();
         while (iterator.hasNext()) {
            sum += iterator.next().getValue();
         }
         return sum;
      }).sum();
   }

   public NDArray sum(Axis axis) {
      Validation.checkArgument(axis == Axis.ROW || axis == Axis.COLUMN,
                               "Only ROW and Axis.COLUMN supported");
      int[] newShape = shape();
      newShape[axis.T().ordinal] = 1;
      NDArray out = getFactory().zeros(newShape);

      sliceStream().forEach(t -> {
         NDArray outSlice = out.slice(t.v1);
         t.v2.iterator().forEachRemaining(e -> {
            int i = axis.select(e.row, e.column);
            outSlice.set(i, outSlice.get(i) + e.getValue());
         });
      });

      return out;
   }

   public float sumOfSquares() {
      return (float) sliceStream().mapToDouble(t ->
                                                  Streams.asStream(t.v2)
                                                         .mapToDouble(e -> Math.pow(e.getValue(), 2))
                                                         .sum()
                                              ).sum();
   }

   public NDArray test(DoublePredicate predicate) {
      return mapOperator(newZeroArray(), v -> predicate.test(v) ? 1 : 0f);
   }

   public NDArray testi(DoublePredicate predicate) {
      return mapOperator(this, v -> predicate.test(v) ? 1 : 0f);
   }

   public abstract DoubleMatrix toDoubleMatrix();

   public float[] toFloatArray() {
      float[] out = new float[(int) length()];
      iterator().forEachRemaining(e -> out[(int) Util.index(e.row, dimension(Axis.ROW),
                                                            e.column, dimension(Axis.COLUMN),
                                                            e.kernel, dimension(Axis.KERNEL),
                                                            e.channel, dimension(Axis.CHANNEL)
                                                           )] = e.getValue());
      return out;
   }

   public abstract FloatMatrix toFloatMatrix();

   private int toIndex(int ax1, int dimAx1, int ax2, int dimAx2) {
      return ax1 + (dimAx1 * ax2);
   }

   private int toIndex(int[] indices, Axis ax1, Axis ax2) {
      return toIndex(indices[ax1.ordinal], shape[ax1.ordinal],
                     indices[ax2.ordinal], shape[ax2.ordinal]);
   }

   protected long toIndex(int[] indices) {
      int sliceIndex = toIndex(indices, Axis.KERNEL, Axis.CHANNEL);
      int matrixIndex = toIndex(indices, Axis.ROW, Axis.COLUMN);
      int matrixLength = shape[Axis.ROW.ordinal] * shape[Axis.COLUMN.ordinal];
      int sliceLength = shape[Axis.KERNEL.ordinal] * shape[Axis.CHANNEL.ordinal];
      return toIndex(matrixIndex, matrixLength, sliceIndex, sliceLength);
   }

   public NDArray neg() {
      return map(v -> -v);
   }

   public NDArray negi() {
      return mapi(v -> -v);
   }

   public NDArray pow(int power) {
      return map(v -> FastMath.pow(v, power));
   }

   public NDArray powi(int power) {
      return mapi(v -> FastMath.pow(v, power));
   }

   public NDArray exp() {
      return map(FastMath::exp);
   }

   public NDArray expi() {
      return mapi(FastMath::exp);
   }

   public NDArray log() {
      return map(Math2::safeLog);
   }

   public NDArray logi() {
      return mapi(Math2::safeLog);
   }


   @Override
   public JsonEntry toJson() {
      JsonEntry ndarray = JsonEntry.object()
                                   .addProperty("shape", shape())
                                   .addProperty("dense", isDense());
      JsonEntry array = JsonEntry.array();
      for (int i = 0; i < slices(); i++) {
         array.addValue(slice(i).toFloatArray());
      }
      ndarray.addProperty("data", array);
      return ndarray;
   }

   public float[] toMatrixArray() {
      checkArgument(isMatrix(), "Order (" + order + ") not supported");
      float[] out = new float[matrixLength];
      sparseIterator().forEachRemaining(e -> {
         out[e.matrixIndex()] = e.getValue();
      });
      return out;
   }

   public float[][] toTensorArray() {
      float[][] out = new float[numSlices][matrixLength];
      sparseIterator().forEachRemaining(e -> {
         out[e.sliceIndex()][e.matrixIndex()] = e.getValue();
      });
      return out;
   }

   protected class IndicesIterator implements Iterator<int[]> {
      private int[] indices = new int[4];

      @Override
      public boolean hasNext() {
         return indices[0] < shape[0]
                   && indices[1] < shape[1]
                   && indices[2] < shape[2]
                   && indices[3] < shape[3];
      }

      private void incrementChannel() {
         indices[3]++;
         if (indices[3] >= shape[3]) {
            indices[3] = 0;
            incrementKernel();
         }
      }

      private void incrementColumn() {
         indices[1]++;
         if (indices[1] >= shape[1]) {
            indices[1] = 0;
            incrementRow();
         }
      }

      private void incrementKernel() {
         indices[2]++;
      }

      private void incrementRow() {
         indices[0]++;
         if (indices[0] >= shape[0]) {
            indices[0] = 0;
            incrementChannel();
         }
      }

      @Override
      public int[] next() {
         checkArgument(hasNext(), "No next index");
         int[] next = new int[]{indices[0], indices[1], indices[2], indices[3]};
         incrementColumn();
         return next;
      }

   }

   public class Entry {
      final int row, column, kernel, channel;

      protected Entry(int[] indices) {
         this(indices[0], indices[1], indices[2], indices[3]);
      }

      protected Entry(int row, int column, int kernel, int channel) {
         this.row = row;
         this.column = column;
         this.kernel = kernel;
         this.channel = channel;
      }

      @Override
      public boolean equals(Object obj) {
         if (this == obj) {return true;}
         if (obj == null || getClass() != obj.getClass()) {return false;}
         final Entry other = (Entry) obj;
         return Objects.equals(this.row, other.row)
                   && Objects.equals(this.column, other.column)
                   && Objects.equals(this.kernel, other.kernel)
                   && Objects.equals(this.channel, other.channel);
      }

      public int getChannel() {
         return channel;
      }

      public int getColumn() {
         return column;
      }

      public long getIndex() {
         return toIndex(new int[]{row, column, kernel, channel});
      }

      public int getIndex(Axis axis) {
         switch (axis) {
            case ROW:
               return getRow();
            case COLUMN:
               return getColumn();
            case KERNEL:
               return getKernel();
            case CHANNEL:
               return getChannel();
            default:
               throw new IllegalArgumentException(axis + " is an unkown axis");
         }
      }

      public int[] getIndicies() {
         return new int[]{
            getRow(),
            getColumn(),
            getKernel(),
            getChannel()
         };
      }

      public int getKernel() {
         return kernel;
      }

      public int getRow() {
         return row;
      }

      public float getValue() {
         return get(row, column, kernel, channel);
      }

      public void setValue(float value) {
         set(row, column, kernel, channel, value);
      }

      @Override
      public int hashCode() {
         return Objects.hash(row, column, kernel, channel);
      }

      public int matrixIndex() {
         return toIndex(row, shape[0], column, shape[1]);
      }

      public int sliceIndex() {
         return toIndex(kernel, shape[2], channel, shape[3]);
      }

      @Override
      public String toString() {
         return "Entry[" +
                   "row=" + row +
                   ", column=" + column +
                   ", kernel=" + kernel +
                   ", channel=" + channel +
                   "]=" + getValue();
      }

   }

}//END OF NDArray
