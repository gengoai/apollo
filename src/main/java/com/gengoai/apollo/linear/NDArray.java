package com.gengoai.apollo.linear;

import com.gengoai.Copyable;
import com.gengoai.Validation;
import com.gengoai.collection.Iterators;
import com.gengoai.collection.Streams;
import com.gengoai.conversion.Cast;
import com.gengoai.json.JsonEntry;
import com.gengoai.json.JsonSerializable;
import com.gengoai.math.Math2;
import com.gengoai.math.Operator;
import com.gengoai.math.Optimum;
import com.gengoai.tuple.Tuple2;
import lombok.NonNull;
import org.apache.commons.math3.util.FastMath;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;
import java.util.function.*;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.checkElementIndex;
import static com.gengoai.apollo.linear.NDArrayFactory.DENSE;
import static com.gengoai.collection.Iterators.zipWithIndex;
import static com.gengoai.tuple.Tuples.$;

/**
 * The type Nd array.
 *
 * @author David B. Bracewell
 */
public abstract class NDArray implements Copyable<NDArray>, Serializable, JsonSerializable, Iterable<NDArray.Entry> {
   private static final long serialVersionUID = 1L;
   /**
    * The Shape.
    */
   protected final int[] shape;
   private final long length;
   private final int matrixLength;
   private final int numSlices;
   private int order;
   private Object label;
   private Object predicted;
   private float weight;


   public boolean shapeEquals(NDArray other) {
      return Arrays.equals(shape, other.shape);
   }

   @Override
   public int hashCode() {
      return Objects.hash(label, predicted, weight, toFloatArray());
   }

   @Override
   public boolean equals(Object o) {
      if (o instanceof NDArray) {
         NDArray oa = Cast.as(o);
         if (Objects.equals(label, oa.label) &&
                Objects.equals(predicted, oa.predicted) &&
                Objects.equals(weight, oa.weight) &&
                shapeEquals(oa)) {
            for (int i = 0; i < slices(); i++) {
               if (!Arrays.equals(tensorSlice(i).toFloatArray(), oa.tensorSlice(i).toFloatArray())) {
                  return false;
               }
            }
            return true;
         }
         return false;
      }
      return false;
   }


   /**
    * Zero nd array.
    *
    * @return the nd array
    */
   public NDArray zero() {
      return fill(0f);
   }

   /**
    * Default Constructor
    *
    * @param shape The shape of the new NDArray
    * @throws IllegalArgumentException if the length of the shape array is greater than four.
    */
   protected NDArray(int[] shape) {
      checkArgument(shape.length <= 4, "Invalid shape of length (" + shape.length + ").");
      this.shape = new int[]{1, 1, 1, 1};
      System.arraycopy(shape, 0, this.shape, 0, shape.length);
      this.order = Arrays.stream(shape).map(i -> i > 1 ? 1 : 0).sum();
      this.matrixLength = this.shape[0] * this.shape[1];
      this.numSlices = this.shape[2] * this.shape[3];
      this.length = this.matrixLength * this.numSlices;
   }

   /**
    * Ensure correct indices int [ ].
    *
    * @param dimensions the dimensions
    * @return the int [ ]
    */
   public static int[] ensureCorrectIndices(int... dimensions) {
      int[] shape = new int[]{1, 1, 1, 1};
      System.arraycopy(dimensions, 0, shape, 0, dimensions.length);
      return shape;
   }

   /**
    * From index int [ ].
    *
    * @param index  the index
    * @param dimAx1 the dim ax 1
    * @param dimAx2 the dim ax 2
    * @return the int [ ]
    */
   public static int[] fromIndex(long index, int dimAx1, int dimAx2) {
      return new int[]{
         (int) index % dimAx1,
         (int) index / dimAx1
      };
   }

   /**
    * From index int [ ].
    *
    * @param index the index
    * @param shape the shape
    * @return the int [ ]
    */
   public static int[] fromIndex(long index, int[] shape) {
      shape = ensureCorrectIndices(shape);
      int matrixLength = shape[0] * shape[1];
      int sliceLength = shape[2] * shape[3];
      int[] imd = fromIndex(index, matrixLength, sliceLength);
      int[] matrix = fromIndex(imd[0], shape[0], shape[1]);
      int[] slice = fromIndex(imd[1], shape[2], shape[3]);
      return new int[]{
         matrix[0],
         matrix[1],
         slice[0],
         slice[1]
      };
   }

   /**
    * Method for constructing an NDArray from JsonEntry for use in serializing / deserializing to/from json.
    *
    * @param entry The <code>JsonEntry</code> to parse the NDArray from
    * @return The NDArray from the JsonEntry
    */
   public static NDArray fromJson(JsonEntry entry) {
      if (entry.getBooleanProperty("dense")) {
         return DenseNDArray.fromJson(entry);
      }
      NDArray ndArray = DENSE.zeros(entry.getValProperty("shape")
                                         .asIntegerValueArray());
      JsonEntry array = entry.getProperty("data");
      zipWithIndex(array.elementIterator()).forEachRemaining(e -> {
         NDArray matrix = ndArray.tensorSlice(e.getValue());
         zipWithIndex(e.getKey().elementIterator())
            .forEachRemaining(v -> matrix.set(v.getValue(), v.getKey().getAsFloat()));
      });
      return ndArray;
   }

   /**
    * To index int.
    *
    * @param ax1    the ax 1
    * @param dimAx1 the dim ax 1
    * @param ax2    the ax 2
    * @param dimAx2 the dim ax 2
    * @return the int
    */
   public static int toIndex(int ax1, int dimAx1, int ax2, int dimAx2) {
      return ax1 + (dimAx1 * ax2);
   }

   protected int toSliceIndex(int kernel, int channel) {
      return toIndex(kernel, shape[Axis.KERNEL.ordinal],
                     channel, shape[Axis.CHANNEL.ordinal]);
   }

   protected int toMatrixIndex(int row, int column) {
      return toIndex(row, shape[Axis.ROW.ordinal],
                     column, shape[Axis.COLUMN.ordinal]);
   }

   /**
    * To long index long.
    *
    * @param indices the indices
    * @param shape   the shape
    * @return the long
    */
   public static long toLongIndex(int[] indices, int[] shape) {
      checkArgument(indices.length <= 4,
                    "Invalid number of indices (" + indices.length + ")");
      switch (indices.length) {
         case 0:
            return 0;
         case 1:
            return indices[0];
         case 2:
            return toIndex(indices[0], shape[0], indices[1], shape[1]);
         case 3:
            return toIndex(toIndex(indices[0], shape[0], indices[1], shape[1]),
                           shape[0] * shape[1],
                           indices[2],
                           shape[2]);
         case 4:
            return toIndex(toIndex(indices[0], shape[0], indices[1], shape[1]),
                           shape[0] * shape[1],
                           toIndex(indices[2], shape[2], indices[3], shape[3]),
                           shape[2] * shape[3]);
      }
      throw new IllegalArgumentException("Invalid number of indices (" + indices.length + ")");
   }

   /**
    * Flips the matrix on its diagonal switching the rows and columns. (This is done per slice)
    *
    * @return the transposed array
    */
   public NDArray T() {
      return sliceUnaryOperation(v -> {
         NDArray out = getFactory().zeros(v.numCols(), v.numRows());
         v.forEach(e -> out.set(e.getColumn(), e.getRow(), e.getValue()));
         return out;
      });
   }

   /**
    * Takes the absolute value of the values in the NDArray
    *
    * @return the new NDArray with absolute values
    */
   public NDArray abs() {
      return map(Math::abs);
   }

   /**
    * Takes the absolute value of the values in the NDArray in-place
    *
    * @return this NDArray
    */
   public NDArray absi() {
      return mapi(Math::abs);
   }

   /**
    * Adds a scalar value to each element in the NDArray
    *
    * @param scalar the value to add
    * @return the new NDArray with the scalar value added
    */
   public NDArray add(double scalar) {
      return mapScalar(newZeroArray(),
                       scalar,
                       Operator::add);
   }

   /**
    * Adds the values in the other NDArray to this one. Basic broadcasting will occur for scalar, vector, and matrix
    * NDArrays.
    *
    * @param other the other NDArray whose values will be added
    * @return the new NDArray with the result of this + other
    */
   public NDArray add(NDArray other) {
      return map(newZeroArray(),
                 other,
                 Operator::add);
   }

   /**
    * Performs a column or row vector addition adding the values in the other NDArray to each row or column in this
    * NDArray as specified by the given axis parameter.
    *
    * @param other the other NDArray whose values will be added
    * @param axis  the axis
    * @return the new NDArray with the result of this + other
    */
   public NDArray add(NDArray other, Axis axis) {
      return mapVector(newZeroArray(),
                       other,
                       axis,
                       Operator::add);
   }

   /**
    * Adds a scalar value to each element in the NDArray in-place
    *
    * @param scalar the value to add
    * @return this NDArray with the scalar value added
    */
   public NDArray addi(double scalar) {
      return mapScalar(this,
                       scalar,
                       Operator::add);
   }

   /**
    * Adds the values in the other NDArray to this one in-place.  Basic broadcasting will occur for scalar, vector, and
    * matrix NDArrays.
    *
    * @param other the other NDArray whose values will be added
    * @return this NDArray with the result of this + other
    */
   public NDArray addi(NDArray other) {
      return map(this,
                 other,
                 Operator::add);
   }

   /**
    * Performs a column or row vector addition adding the values in the other NDArray to each row or column in this
    * NDArray as specified by the given axis parameter in-place.
    *
    * @param other the other NDArray whose values will be added
    * @param axis  the axis
    * @return this NDArray with the result of this + other
    */
   public NDArray addi(NDArray other, Axis axis) {
      return mapVector(this,
                       other,
                       axis,
                       Operator::add);
   }

   /**
    * Calculates the index of the maximum along each row or column based on the given axis.
    *
    * @param axis the axis (row/column) to calculate the max for
    * @return array of int array of row/column indexes relating to max values per slice
    */
   public int[][] argMax(Axis axis) {
      return argOptimum(axis, Optimum.MAXIMUM);
   }

   /**
    * Calculates the index of the minimum along each row or column based on the given axis.
    *
    * @param axis the axis (row/column) to calculate the minimum for
    * @return array of int array of row/column indexes relating to minimum values per slice
    */
   public int[][] argMin(Axis axis) {
      return argOptimum(axis, Optimum.MINIMUM);
   }

   private int[][] argOptimum(Axis axis, Optimum optimum) {
      checkArgument(axis.isRowOrColumn(), "Axis (" + axis + ") not supported");
      int[][] out = new int[numSlices][dimension(axis)];
      double[][] optimums = new double[numSlices][dimension(axis)];
      for (int slice = 0; slice < numSlices; slice++) {
         Arrays.fill(optimums[slice], optimum.startingValue());
      }
      forEach(e -> {
         if (optimum.test(e.getValue(), optimums[e.sliceIndex()][e.getIndex(axis)])) {
            optimums[e.sliceIndex()][e.getIndex(axis)] = e.getValue();
            out[e.sliceIndex()][e.getIndex(axis)] = e.getIndex(axis.T());
         }
      });
      return out;
   }

   /**
    * Number of Channels in the NDArray
    *
    * @return the number of channels
    */
   public int numChannels() {
      return dimension(Axis.CHANNEL);
   }

   /**
    * Number of columns in the NDArray
    *
    * @return the number of columns
    */
   public int numCols() {
      return dimension(Axis.COLUMN);
   }

   /**
    * Compresses the memory used by the NDArray. (Only useful for sparse implementations)
    *
    * @return this NDArray
    */
   public NDArray compress() {
      return this;
   }

   /**
    * Calculates the variance-covariance matrix of this NDArray. Each slice is operated on independently.
    *
    * @return The variance-covariance NDArray
    */
   public NDArray cov() {
      return sliceUnaryOperation(v -> {
         NDArray c = v.sub(getFactory().ones(v.numRows(), v.numRows())
                                       .mmul(v)
                                       .muli(1f / v.numRows()));
         return c.T().mmul(c).divi(v.numRows());
      });
   }

   /**
    * Decrements the value of the NDArray at the given indices.
    *
    * @param indices the indices of the value to decrement
    * @param value   the value to decrement by
    * @return this NDArray
    */
   public NDArray decrement(int[] indices, double value) {
      return set(indices, get(indices) - value);
   }

   /**
    * Decrements the value of the NDArray at the given indices.
    *
    * @param row     the row
    * @param column  the column
    * @param kernel  the kernel
    * @param channel the channel
    * @param value   the value to decrement by
    * @return This NDArray
    */
   public NDArray decrement(int row, int column, int kernel, int channel, double value) {
      return set(row, column, kernel, channel, get(row, column, kernel, channel) - value);
   }

   /**
    * Decrements the value of the NDArray at the given indices.
    *
    * @param row    the row
    * @param column the column
    * @param kernel the kernel
    * @param value  the value to decrement by
    * @return This NDArray
    */
   public NDArray decrement(int row, int column, int kernel, double value) {
      return set(row, column, kernel, get(row, column, kernel) - value);
   }

   /**
    * Decrements the value of the NDArray at the given indices.
    *
    * @param row    the row
    * @param column the column
    * @param value  the value to decrement by
    * @return This NDArray
    */
   public NDArray decrement(int row, int column, double value) {
      return set(row, column, get(row, column) - value);
   }

   /**
    * Decrements the value of the NDArray at the given indices.
    *
    * @param row   the row
    * @param value the value to decrement by
    * @return This NDArray
    */
   public NDArray decrement(int row, double value) {
      return set(row, get(row) - value);
   }

   /**
    * Generates a diagonal matrix per slice.
    *
    * @return The NDArray with diagonal slices.
    */
   public NDArray diag() {
      return sliceUnaryOperation(v -> {
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
            NDArray out = getFactory().zeros(v.numRows(), v.numCols());
            for (int i = 0; i < v.numRows(); i++) {
               if (i < v.numCols()) {
                  out.set(i, i, v.get(i, i));
               }
            }
            return out;
         }
         throw new IllegalStateException("Rectangular slices are not supported");
      });
   }

   /**
    * Gets the dimension of the the given axis
    *
    * @param axis the axis
    * @return the dimension of the given axis
    */
   public int dimension(Axis axis) {
      return shape[axis.ordinal];
   }

   /**
    * Divides the values in the other NDArray to this one element by element. Basic broadcasting will occur for scalar,
    * vector, and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be divided
    * @return the new NDArray with the result of this / other
    */
   public NDArray div(NDArray other) {
      return mapSparse(newZeroArray(),
                       other,
                       Operator::divide);
   }

   /**
    * Divides a scalar value to each element in the NDArray
    *
    * @param value the value to divide
    * @return the new NDArray with the scalar value divided
    */
   public NDArray div(double value) {
      return mapSparseScalar(newZeroArray(),
                             value,
                             Operator::divide);
   }

   /**
    * Divides a column or row vector element division dividing the values in the other NDArray to each row or column in
    * this NDArray as specified by the given axis parameter.
    *
    * @param other the other NDArray whose values will be divided
    * @param axis  the axis
    * @return the new NDArray with the result of this / other
    */
   public NDArray div(NDArray other, Axis axis) {
      return mapSparseVector(newZeroArray(),
                             other,
                             axis,
                             Operator::divide);
   }

   /**
    * Divides a scalar value to each element in the NDArray in-place.
    *
    * @param value the value to divide
    * @return this NDArray with the scalar value divided
    */
   public NDArray divi(double value) {
      return mapSparseScalar(this,
                             value,
                             Operator::divide);
   }

   /**
    * Divides the values in the other NDArray to this one element by element in-place. Basic broadcasting will occur for
    * scalar, vector, and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be divided
    * @return this NDArray with the result of this / other
    * @throws IllegalArgumentException If the shape of this NDArray does not match that of the other NDArray
    */
   public NDArray divi(NDArray other) {
      return mapSparse(this,
                       other,
                       Operator::divide);
   }

   /**
    * Divides a column or row vector element division dividing the values in the other NDArray to each row or column in
    * this NDArray as specified by the given axis parameter in-place.
    *
    * @param other the other NDArray whose values will be divided
    * @param axis  the axis
    * @return this NDArray with the result of this / other
    */
   public NDArray divi(NDArray other, Axis axis) {
      return mapSparseVector(this,
                             other,
                             axis,
                             Operator::divide);
   }

   /**
    * Calculates the dot product of vectors along the given axis between this NDArray and the given NDArray. Operation
    * will be repeated across slices.
    *
    * @param other the other NDArray to calculate the dot product with
    * @param axis  the axis to calculate the dot product along
    * @return The NDArray containing the dot product
    */
   public NDArray dot(NDArray other, Axis axis) {
      checkArgument(axis.isRowOrColumn(), "Axis (" + axis + ") is not supported.");
      checkArgument(dimension(axis.T()) == other.dimension(axis.T()),
                    "Dimension (" + axis.T() + ") mismatch (" + dimension(axis.T()) + ")  != (" + other.dimension(
                       axis.T()) + ")");
      int[] newShape = shape();
      newShape[axis.T().ordinal] = 1;
      NDArray out = getFactory().zeros(newShape);
      sliceStream().forEach(t -> {
         NDArray otherSlice = other.tensorSlice(t.v1);
         NDArray outSlice = out.tensorSlice(t.v1);
         t.v2.forEachSparse(e -> outSlice.increment(e.getIndex(axis),
                                                    otherSlice.get(e.row, e.column) * e.getValue()
                                                   ));
      });
      return out;
   }

   /**
    * Calculates the dot product of vectors between this NDArray and the given NDArray. The dot product is summed across
    * the slices.
    *
    * @param other the other NDArray to calculate the dot product with
    * @return The sum of the dot products across the slices.
    */
   public float dot(NDArray other) {
      checkArgument(matrixLength == other.matrixLength,
                    "Length mismatch (" + matrixLength + ")  != (" + other.matrixLength);
      return (float) sliceStream().mapToDouble(t -> {
         NDArray os = other.tensorSlice(t.v1);
         return Streams.asStream(t.v2.sparseIterator()).mapToDouble(e -> e.getValue() * os.get(e.matrixIndex())).sum();
      }).sum();
   }

   /**
    * Calculates <code>exp(v)</code> for each value <code>v</code> in the NDArray
    *
    * @return NDArray with exp values
    */
   public NDArray exp() {
      return map(FastMath::exp);
   }

   /**
    * Calculates <code>exp(v)</code> for each value <code>v</code> in the NDArray updating itself
    *
    * @return This NDArray with exp values
    */
   public NDArray expi() {
      return mapi(FastMath::exp);
   }

   /**
    * Fills the NDArray with the given value
    *
    * @param value the value to set all cells in the NDArray
    * @return This NDArray
    */
   public NDArray fill(double value) {
      forEachSlice(ndarray -> ndarray.forEach(e -> e.setValue(value)));
      return this;
   }

   /**
    * Fills the NDArray with values generated from the given double supplier
    *
    * @param supplier the supplier to use to set all cells in the NDArray
    * @return This NDArray
    */
   public NDArray fill(DoubleSupplier supplier) {
      forEachSlice(ndarray -> ndarray.forEach(e -> e.setValue((float) supplier.getAsDouble())));
      return this;
   }

   /**
    * Processes each slice of the NDArray using the given NDArray consumer
    *
    * @param sliceConsumer the slice consumer
    */
   public void forEachSlice(Consumer<NDArray> sliceConsumer) {
      sliceStream().forEach(t -> sliceConsumer.accept(t.v2));
   }

   /**
    * Processes the non-zero entries in this NDArray using the given entry consumer
    *
    * @param consumer the consumer to use to process non-zero entries
    */
   public void forEachSparse(Consumer<Entry> consumer) {
      sparseIterator().forEachRemaining(consumer);
   }

   /**
    * Gets the value of entry at the given indices
    *
    * @param indices the indices of the value to get
    * @return the value at the given indices
    */
   public abstract float get(int... indices);

   /**
    * Gets the factory used to create NDArrays of this type
    *
    * @return the NDArrayFactory
    */
   public abstract NDArrayFactory getFactory();

   /**
    * Gets label.
    *
    * @param <T> the type parameter
    * @return the label
    */
   public <T> T getLabel() {
      return Cast.as(label);
   }

   /**
    * Sets label.
    *
    * @param label the label
    * @return the label
    */
   public NDArray setLabel(Object label) {
      this.label = label;
      return this;
   }

   /**
    * Gets label as double.
    *
    * @return the label as double
    */
   public float getLabelAsFloat() {
      if (label == null) {
         return Float.NaN;
      }
      return Cast.<Number>as(label).floatValue();
   }

   public long size() {
      return length;
   }

   /**
    * Gets label as double.
    *
    * @return the label as double
    */
   public double getLabelAsDouble() {
      if (label == null) {
         return Double.NaN;
      }
      return Cast.<Number>as(label).doubleValue();
   }

   /**
    * Gets label as nd array.
    *
    * @return the label as nd array
    */
   public NDArray getLabelAsNDArray() {
      if (label == null) {
         return getFactory().empty();
      } else if (label instanceof Number) {
         return getFactory().constant(Cast.<Number>as(label).floatValue(), 1);
      }
      return Cast.as(label);
   }

   /**
    * Gets label as nd array.
    *
    * @param dimension the dimension
    * @return the label as nd array
    */
   public NDArray getLabelAsNDArray(int dimension) {
      if (label == null) {
         return getFactory().empty();
      } else if (label instanceof Number) {
         return getFactory().zeros(dimension)
                            .set(Cast.<Number>as(label).intValue(), 1f);
      }
      return Cast.as(label);
   }

   /**
    * Gets predicted.
    *
    * @param <T> the type parameter
    * @return the predicted
    */
   public <T> T getPredicted() {
      return Cast.as(predicted);
   }

   /**
    * Sets predicted.
    *
    * @param predicted the predicted
    * @return the predicted
    */
   public NDArray setPredicted(Object predicted) {
      this.predicted = predicted;
      return this;
   }

   /**
    * Gets predicted as double.
    *
    * @return the predicted as double
    */
   public double getPredictedAsDouble() {
      if (predicted == null) {
         return Double.NaN;
      }
      return Cast.<Number>as(predicted).doubleValue();
   }

   /**
    * Gets predicted as nd array.
    *
    * @return the predicted as nd array
    */
   public NDArray getPredictedAsNDArray() {
      if (predicted == null) {
         return getFactory().empty();
      } else if (predicted instanceof Number) {
         return getFactory().constant(Cast.<Number>as(predicted).floatValue(), 1);
      }
      return Cast.as(predicted);
   }

   /**
    * Gets label as nd array.
    *
    * @param dimension the dimension
    * @return the label as nd array
    */
   public NDArray getPredictedAsNDArray(int dimension) {
      if (predicted == null) {
         return getFactory().empty();
      } else if (predicted instanceof Number) {
         return getFactory().zeros(dimension)
                            .set(Cast.<Number>as(predicted).intValue(), 1f);
      }
      return Cast.as(predicted);
   }

   /**
    * Gets the vector(s) at the given index along the given axis. This works across slices.
    *
    * @param index the index of the vector to retrieve
    * @param axis  the axis of the vector
    * @return the NDArray of vectors
    */
   public NDArray getVector(int index, Axis axis) {
      checkArgument(axis.isRowOrColumn(), "Axis (" + axis + ") is invalid.");
      int[] newShape = shape();
      newShape[axis.ordinal] = 1;
      NDArray out = getFactory().zeros(newShape);
      sliceStream().forEach(t -> {
         NDArray slice = out.tensorSlice(t.v1);
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

   /**
    * Gets weight.
    *
    * @return the weight
    */
   public float getWeight() {
      return weight;
   }

   /**
    * Sets weight.
    *
    * @param weight the weight
    * @return the weight
    */
   public NDArray setWeight(double weight) {
      this.weight = (float) weight;
      return this;
   }

   /**
    * Increments the value of the NDArray at the given indices.
    *
    * @param indices the indices of the value to decrement
    * @param value   the value to increment by
    * @return This NDArray
    */
   public NDArray increment(int[] indices, double value) {
      return set(indices, get(indices) + value);
   }

   /**
    * Increments the value of the NDArray at the given indices.
    *
    * @param row     the row
    * @param column  the column
    * @param kernel  the kernel
    * @param channel the channel
    * @param value   the value to increment by
    * @return This NDArray
    */
   public NDArray increment(int row, int column, int kernel, int channel, double value) {
      return set(row, column, kernel, channel, get(row, column, kernel, channel) + value);
   }

   /**
    * Increments the value of the NDArray at the given indices.
    *
    * @param row    the row
    * @param column the column
    * @param kernel the kernel
    * @param value  the value to increment by
    * @return This NDArray
    */
   public NDArray increment(int row, int column, int kernel, double value) {
      return set(row, column, kernel, get(row, column, kernel) + value);
   }

   /**
    * Increments the value of the NDArray at the given indices.
    *
    * @param row    the row
    * @param column the column
    * @param value  the value to increment by
    * @return This NDArray
    */
   public NDArray increment(int row, int column, double value) {
      return set(row, column, get(row, column) + value);
   }

   /**
    * Increments the value of the NDArray at the given indices.
    *
    * @param row   the row
    * @param value the value to increment by
    * @return This NDArray
    */
   public NDArray increment(int row, double value) {
      return set(row, get(row) + value);
   }

   /**
    * Checks if the NDArray is a column vector, i.e. a shape of <code>(?,1,1,1)</code>
    *
    * @return True if the NDArray is a column vector
    */
   public boolean isColumnVector() {
      return isMatrix() && dimension(Axis.ROW) > 1 && dimension(Axis.COLUMN) == 1;
   }

   /**
    * Checks if the NDArray is a dense representation
    *
    * @return True if the NDArray is dense
    */
   public boolean isDense() {
      return false;
   }

   /**
    * Checks if the NDArray is a matrix, i.e. a shape of <code>(?,?,1,1)</code>
    *
    * @return True if the NDArray is a matrix
    */
   public boolean isMatrix() {
      return dimension(Axis.KERNEL) == 1 && dimension(Axis.CHANNEL) == 1;
   }

   /**
    * Checks if the NDArray is a row vector, i.e. a shape of <code>(1,?,1,1)</code>
    *
    * @return True if the NDArray is a row vector
    */
   public boolean isRowVector() {
      return isMatrix() && dimension(Axis.ROW) == 1 && dimension(Axis.COLUMN) > 1;
   }

   /**
    * Checks if the NDArray is a scalar, i.e. a shape of <code>(1,1,1,1)</code>
    *
    * @return True if the NDArray is a scalar
    */
   public boolean isScalar() {
      return dimension(Axis.ROW) == 1 && dimension(Axis.COLUMN) == 1 &&
                dimension(Axis.KERNEL) == 1 && dimension(Axis.CHANNEL) == 1;
   }

   /**
    * Checks if the NDArray is a sparse representation
    *
    * @return True if the NDArray is sparse
    */
   public boolean isSparse() {
      return false;
   }

   /**
    * Checks if the NDArray is square, i.e. a shape of <code>(N,N,1,1)</code>
    *
    * @return True if the NDArray is square
    */
   public boolean isSquare() {
      return isMatrix() && numRows() == numCols();
   }

   /**
    * Checks if the NDArray is a vector (row or column)
    *
    * @return True if the NDArray is a vector
    */
   public boolean isVector() {
      return isRowVector() || isColumnVector();
   }

   @Override
   public Iterator<Entry> iterator() {
      return Iterators.transform(new IndicesIterator(), Entry::new);
   }

   /**
    * Number of kernels in the NDArray
    *
    * @return the number of kernels
    */
   public int numKernels() {
      return dimension(Axis.KERNEL);
   }

   /**
    * The total length (rows * columns * kernels * channels) of the NDArray
    *
    * @return the total length of the NDArray
    */
   public long length() {
      return length;
   }

   /**
    * Calculates <code>log(v)</code> for each value <code>v</code> in the NDArray
    *
    * @return NDArray with exp values
    */
   public NDArray log() {
      return map(Math2::safeLog);
   }

   /**
    * Calculates <code>log(v)</code> for each value <code>v</code> in the NDArray updating itself
    *
    * @return This NDArray with exp values
    */
   public NDArray logi() {
      return mapi(Math2::safeLog);
   }

   /**
    * Performs the given operator on each entry in the NDArray storing the result in a new NDArray
    *
    * @param operator the operation to perform
    * @return The new  NDArray with the operator applied to this NDArray's values
    */
   public NDArray map(DoubleUnaryOperator operator) {
      return mapOperator(newZeroArray(), operator);
   }

   /**
    * Performs the given operator on each sparse entry in the NDArray storing the result in a new NDArray
    *
    * @param operator the operation to perform
    * @return The new  NDArray with the operator applied to this NDArray's values
    */
   public NDArray mapSparse(DoubleUnaryOperator operator) {
      return mapSparseOperator(newZeroArray(), operator);
   }

   /**
    * Performs the given operator on each sparse entry in the NDArray storing the result in a new NDArray
    *
    * @param operator the operation to perform
    * @return The new  NDArray with the operator applied to this NDArray's values
    */
   public NDArray mapSparse(NDArray other, DoubleBinaryOperator operator) {
      return mapSparse(newZeroArray(), other, operator);
   }

   /**
    * Applies an operation to the non-zero elements in this NDArray and given other NDArray using the given operator
    * in-place.
    *
    * @param operator the operator to apply
    * @return this NDArray
    */
   public NDArray mapiSparse(DoubleUnaryOperator operator) {
      return mapSparseOperator(this, operator);
   }


   /**
    * Performs the given operator on each sparse entry in the NDArray storing the result in a new NDArray
    *
    * @param operator the operation to perform
    * @return The new  NDArray with the operator applied to this NDArray's values
    */
   public NDArray mapiSparse(NDArray other, DoubleBinaryOperator operator) {
      return mapSparse(newZeroArray(), other, operator);
   }

   private NDArray mapSparseOperator(NDArray out, DoubleUnaryOperator operator) {
      forEachSparse(e -> out.set(e.getIndicies(), (float) operator.applyAsDouble(e.getValue())));
      return out;
   }

   private NDArray map(NDArray out, NDArray other, DoubleBinaryOperator operator) {
      if (other.isScalar()) {
         return mapScalar(out, other.get(0), operator);
      }
      if (other.isVector() && !isVector()) {
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

   private NDArray mapSparse(NDArray out, NDArray other, DoubleBinaryOperator operator) {
      if (other.isScalar()) {
         return mapSparseScalar(out, other.get(0), operator);
      }
      if (other.isVector() && !isVector()) {
         return mapSparseVector(out,
                                other,
                                other.isRowVector() ? Axis.COLUMN : Axis.ROW,
                                operator);
      }
      if (other.isMatrix()) {
         return mapSparseMatrix(out, other, operator);
      }
      return mapSparseTensor(out, other, operator);
   }

   /**
    * Applies an operation to the elements in this NDArray and given other NDArray using the given operator producing a
    * new NDArray as its outcome. Basic broadcasting will occur for scalar, vector, and matrix NDArrays.
    *
    * @param other    the other NDArray to perform operation over
    * @param operator the operator to apply
    * @return the new NDArray
    */
   public NDArray map(NDArray other, DoubleBinaryOperator operator) {
      return map(newZeroArray(), other, operator);
   }

   /**
    * Applies the given operator to each element in this NDArray and the given vector along the given axis creating a
    * new NDArray in the process.
    *
    * @param other    the vector of values to combine with this NDArray
    * @param axis     the axis to apply the operator to
    * @param operator the operator to apply to the elements in this NDArray and the given vector
    * @return the new NDArray
    */
   public NDArray map(NDArray other, Axis axis, DoubleBinaryOperator operator) {
      return mapVector(newZeroArray(),
                       other,
                       axis,
                       operator);
   }

   private NDArray mapMatrix(NDArray out, NDArray matrix, DoubleBinaryOperator operator) {
      sliceStream().forEach(t -> {
         NDArray s2 = out.tensorSlice(t.v1);
         t.v2.forEach(
            e -> s2.set(e.matrixIndex(), (float) operator.applyAsDouble(e.getValue(), matrix.get(e.matrixIndex()))));
      });
      return out;
   }

   private NDArray mapSparseMatrix(NDArray out, NDArray matrix, DoubleBinaryOperator operator) {
      sliceStream().forEach(t -> {
         NDArray s2 = out.tensorSlice(t.v1);
         t.v2.forEachSparse(
            e -> s2.set(e.matrixIndex(), (float) operator.applyAsDouble(e.getValue(), matrix.get(e.matrixIndex()))));
      });
      return out;
   }

   private NDArray mapOperator(NDArray out, DoubleUnaryOperator operator) {
      sliceStream().forEach(t -> {
         NDArray slice = out.tensorSlice(t.v1);
         t.v2.forEach(e -> slice.set(e.matrixIndex(), (float) operator.applyAsDouble(e.getValue())));
      });
      return out;
   }

   private NDArray mapScalar(NDArray out, double scalar, DoubleBinaryOperator operator) {
      sliceStream().forEach(t -> {
         NDArray s2 = out.tensorSlice(t.v1);
         t.v2.forEach(e -> s2.set(e.matrixIndex(), (float) operator.applyAsDouble(e.getValue(), scalar)));
      });
      return out;
   }

   private NDArray mapSparseScalar(NDArray out, double scalar, DoubleBinaryOperator operator) {
      sliceStream().forEach(t -> {
         NDArray s2 = out.tensorSlice(t.v1);
         t.v2.forEachSparse(e -> s2.set(e.matrixIndex(), (float) operator.applyAsDouble(e.getValue(), scalar)));
      });
      return out;
   }

   private NDArray mapTensor(NDArray out, NDArray tensor, DoubleBinaryOperator operator) {
      checkArgument(tensor.sliceLength() == sliceLength(),
                    "Length of each slice is not the same. (" + sliceLength() + ") != (" + tensor.sliceLength() + ")");
      checkArgument(slices() == tensor.slices(),
                    "Number of slices does not match. (" + slices() + ") != (" + tensor.slices() + ")");
      sliceStream().forEach(t -> {
         NDArray s2 = tensor.tensorSlice(t.v1);
         NDArray sOut = out.tensorSlice(t.v1);
         t.v2.forEach(
            e -> sOut.set(e.matrixIndex(), (float) operator.applyAsDouble(e.getValue(), s2.get(e.matrixIndex()))));
      });
      return out;
   }

   private NDArray mapSparseTensor(NDArray out, NDArray tensor, DoubleBinaryOperator operator) {
      checkArgument(tensor.sliceLength() == sliceLength(),
                    "Length of each slice is not the same. (" + sliceLength() + ") != (" + tensor.sliceLength() + ")");
      checkArgument(slices() == tensor.slices(),
                    "Number of slices does not match. (" + slices() + ") != (" + tensor.slices() + ")");
      sliceStream().forEach(t -> {
         NDArray s2 = tensor.tensorSlice(t.v1);
         NDArray sOut = out.tensorSlice(t.v1);
         t.v2.forEachSparse(
            e -> sOut.set(e.matrixIndex(), (float) operator.applyAsDouble(e.getValue(), s2.get(e.matrixIndex()))));
      });
      return out;
   }

   private NDArray mapVector(NDArray out, NDArray rowVector,
                             Axis axis, DoubleBinaryOperator operator
                            ) {
      sliceStream().forEach(t -> {
         NDArray s2 = out.tensorSlice(t.v1);
         t.v2.forEach(e -> s2.set(e.matrixIndex(), (float) operator.applyAsDouble(e.getValue(), rowVector.get(
            axis.T().select(e.row, e.column)))));
      });
      return out;
   }

   private NDArray mapSparseVector(NDArray out, NDArray rowVector,
                                   Axis axis, DoubleBinaryOperator operator
                                  ) {
      sliceStream().forEach(t -> {
         NDArray s2 = out.tensorSlice(t.v1);
         t.v2.forEachSparse(e -> s2.set(e.matrixIndex(), (float) operator.applyAsDouble(e.getValue(), rowVector.get(
            axis.T().select(e.row, e.column)))));
      });
      return out;
   }

   /**
    * Applies an operation to the elements in this NDArray and given other NDArray using the given operator updating
    * itself as its outcome. Basic broadcasting will occur for scalar, vector, and matrix NDArrays.
    *
    * @param other    the other NDArray to perform operation over
    * @param operator the operator to apply
    * @return this NDArray
    */
   public NDArray mapi(NDArray other, DoubleBinaryOperator operator) {
      return map(this, other, operator);
   }

   /**
    * Applies an operation to the elements in this NDArray and given other NDArray using the given operator in-place.
    *
    * @param operator the operator to apply
    * @return this NDArray
    */
   public NDArray mapi(DoubleUnaryOperator operator) {
      return mapOperator(this, operator);
   }

   /**
    * Applies the given operator to each element in this NDArray and the given vector along the given axis updating
    * itself.
    *
    * @param other    the vector of values to combine with this NDArray
    * @param axis     the axis to apply the operator to
    * @param operator the operator to apply to the elements in this NDArray and the given vector
    * @return this NDArray
    */
   public NDArray mapi(NDArray other, Axis axis, DoubleBinaryOperator operator) {
      return mapVector(this,
                       other,
                       axis,
                       operator);
   }

   /**
    * Finds the max value in the NDArray
    *
    * @return the max value
    */
   public float max() {
      return optimum(Optimum.MAXIMUM);
   }

   /**
    * Creates a new NDArray with max values across the given access. Works on a per slice basis
    *
    * @param axis the axis we want max values for
    * @return NDArray of maxes
    */
   public NDArray max(Axis axis) {
      return optimum(axis, Optimum.MAXIMUM);
   }

   /**
    * Finds the mean of the values in the NDArray
    *
    * @return the mean value
    */
   public float mean() {
      return sum() / length;
   }

   /**
    * Creates a new NDArray with mean values across the given access. Works on a per slice basis
    *
    * @param axis the axis we want mean values for
    * @return NDArray of mean
    */
   public NDArray mean(Axis axis) {
      return sum(axis).divi(dimension(axis.T()));
   }

   /**
    * Finds the min value in the NDArray
    *
    * @return the min value
    */
   public float min() {
      return optimum(Optimum.MINIMUM);
   }

   /**
    * Creates a new NDArray with min values across the given access. Works on a per slice basis
    *
    * @param axis the axis we want min values for
    * @return NDArray of min
    */
   public NDArray min(Axis axis) {
      return optimum(axis, Optimum.MINIMUM);
   }

   public Iterator<Entry> sparseRowIterator(final int row) {
      checkArgument(order <= 2, "Order (" + order + ") is not supported.");
      return Iterators.filter(rowIterator(row), e -> e.getValue() != 0);
   }

   public Iterator<Entry> rowIterator(final int row) {
      checkArgument(order <= 2, "Order (" + order + ") is not supported.");
      return new Iterator<Entry>() {
         int column = 0;

         @Override
         public boolean hasNext() {
            return column < numCols();
         }

         @Override
         public Entry next() {
            checkElementIndex(column, numCols());
            Entry e = new Entry(row, column, 0, 0);
            column++;
            return e;
         }
      };
   }

   public Iterator<Entry> sparseColumnIterator(final int column) {
      checkArgument(order <= 2, "Order (" + order + ") is not supported.");
      return Iterators.filter(columnIterator(column), e -> e.getValue() != 0);
   }

   public Iterator<Entry> columnIterator(final int column) {
      checkArgument(order <= 2, "Order (" + order + ") is not supported.");
      return new Iterator<Entry>() {
         int row = 0;

         @Override
         public boolean hasNext() {
            return row < numRows();
         }

         @Override
         public Entry next() {
            checkElementIndex(row, numRows());
            Entry e = new Entry(row, column, 0, 0);
            row++;
            return e;
         }
      };
   }

   /**
    * Calculates the product of this and the given NDArray (i.e. matrix multiplication). Works on a per slice basis
    *
    * @param other The other NDArray to multiple
    * @return a new NDArray that is the result of this X other
    */
   public NDArray mmul(NDArray other) {
      if (order <= 2 && other.order <= 2) {
         NDArray toReturn = getFactory().zeros(numRows(), other.numCols());
         return mmul(toReturn, other);
      }
      NDArray out = getFactory().zeros(numRows(),
                                       other.numCols(),
                                       Math.max(numKernels(), other.numKernels()),
                                       Math.max(numChannels(), other.numChannels()));
      if (other.order <= 2) {
         sliceStream().forEach(t -> mmul(out.tensorSlice(t.v1), other));
      } else {
         sliceStream().forEach(t -> mmul(out.tensorSlice(t.v1), other.tensorSlice(t.v1)));
      }
      return out;
   }

   private NDArray mmul(NDArray out, NDArray other) {
      forEach(e -> other.sparseRowIterator(e.column)
                        .forEachRemaining(e2 -> out.increment(e.row, e2.column, e2.getValue() * e.getValue())));
      return out;
   }

   /**
    * Multiplies the values in the other NDArray to this one element by element. Basic broadcasting will occur for
    * scalar, vector, and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be multiplied
    * @return the new NDArray with the result of this * other
    */
   public NDArray mul(NDArray other) {
      return mapSparse(newZeroArray(), other, Operator::multiply);
   }

   /**
    * Multiplies a scalar value to each element in the NDArray
    *
    * @param value the value to multiplied
    * @return the new NDArray with the scalar value multiplied
    */
   public NDArray mul(double value) {
      return mapSparseScalar(newZeroArray(), value, Operator::multiply);
   }

   /**
    * Performs a column or row vector element multiplication multiplying the values in the other NDArray to each row or
    * column in this NDArray as specified by the given axis parameter.
    *
    * @param other the other NDArray whose values will be multiplied
    * @param axis  the axis
    * @return the new NDArray with the result of this * other
    */
   public NDArray mul(NDArray other, Axis axis) {
      return mapSparseVector(newZeroArray(),
                             other,
                             axis,
                             Operator::multiply);
   }

   /**
    * Multiplies a scalar value to each element in the NDArray in-place.
    *
    * @param value the value to multiplied
    * @return this NDArray with the scalar value multiplied
    */
   public NDArray muli(double value) {
      return mapSparseScalar(this, value, Operator::multiply);
   }

   /**
    * Multiplies the values in the other NDArray to this one element by element in-place. Basic broadcasting will occur
    * for scalar, vector, and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be multiplied
    * @return this NDArray with the result of this * other
    */
   public NDArray muli(NDArray other) {
      return mapSparse(this,
                       other,
                       Operator::multiply);
   }

   /**
    * Performs a column or row vector element multiplication multiplying the values in the other NDArray to each row or
    * column in this NDArray as specified by the given axis parameter in-place.
    *
    * @param other the other NDArray whose values will be multiplied
    * @param axis  the axis
    * @return this NDArray with the result of this * other
    */
   public NDArray muli(NDArray other, Axis axis) {
      return mapSparseVector(this,
                             other,
                             axis,
                             Operator::multiply);
   }

   /**
    * Negates the values in the NDArray
    *
    * @return the new NDArray with negated values
    */
   public NDArray neg() {
      return map(v -> -v);
   }

   /**
    * Negates the values in the NDArray in-place
    *
    * @return this NDArray
    */
   public NDArray negi() {
      return mapi(v -> -v);
   }

   private NDArray newZeroArray() {
      return getFactory().zeros(shape)
                         .setWeight(weight)
                         .setLabel(label)
                         .setPredicted(predicted);
   }

   /**
    * Calculates the L1-norm of the NDArray across all slices. See {@link #sliceNorm1()} to calculate per slice.
    *
    * @return the L1-norm
    */
   public float norm1() {
      return (float) sliceStream().mapToDouble(t ->
                                                  Streams.asStream(t.v2)
                                                         .mapToDouble(e -> Math.abs(e.getValue()))
                                                         .sum()
                                              ).sum();
   }

   /**
    * Calculates the L2-norm (magnitude) of the NDArray across all slices. See {@link #sliceNorm2()} to calculate per
    * slice.
    *
    * @return the L1-norm
    */
   public float norm2() {
      return (float) Math.sqrt(sumOfSquares());
   }

   private NDArray optimum(Axis axis, Optimum optimum) {
      Validation.checkArgument(axis == Axis.ROW || axis == Axis.COLUMN,
                               "Only ROW and Axis.COLUMN supported");
      int[] newShape = shape();
      newShape[axis.T().ordinal] = 1;
      NDArray out = getFactory().constant((float) optimum.startingValue(), newShape);
      sliceStream().forEach(t -> {
         NDArray outSlice = out.tensorSlice(t.v1);
         t.v2.iterator().forEachRemaining(e -> {
            int i = axis.select(e.row, e.column);
            if (optimum.test(e.getValue(), outSlice.get(i))) {
               outSlice.set(i, e.getValue());
            }
         });
      });

      return out;
   }

   private float optimum(Optimum optimum) {
      DoubleStream values = sliceStream().mapToDouble(t -> {
         double opt = optimum.startingValue();
         for (Entry aV2 : t.v2) {
            float v = aV2.getValue();
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

   /**
    * Calculates the order (number of dimensions) of the NDArray
    *
    * @return the order of the NDArray
    */
   public int order() {
      return order;
   }

   /**
    * Calculates the pivot elements for this square matrix. Will calculate per slice.
    *
    * @return A NDArray of 1's and 0's representing pivot elements.
    */
   public NDArray pivot() {
      return sliceUnaryOperation(ndArray -> {
         if (ndArray.isSquare()) {
            NDArray p = getFactory().eye(ndArray.numRows());
            for (int i = 0; i < ndArray.numRows(); i++) {
               double max = ndArray.get(i, i);
               int row = i;
               for (int j = i; j < ndArray.numRows(); j++) {
                  if (ndArray.get(j, i) > max) {
                     max = ndArray.get(j, i);
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
         }
         throw new IllegalArgumentException("Only square slices supported");
      });
   }

   /**
    * Raises the value of each element in the NDArray by the given power.
    *
    * @param power the power to raise values to
    * @return the new NDArray
    */
   public NDArray pow(double power) {
      return map(v -> FastMath.pow(v, power));
   }

   /**
    * Raises the value of each element in the NDArray by the given power in-place.
    *
    * @param power the power to raise values to
    * @return this NDArray
    */
   public NDArray powi(double power) {
      return mapi(v -> FastMath.pow(v, power));
   }

   /**
    * Divides the values in the this NDArray from the other NDArray. Basic broadcasting will occur for scalar, vector,
    * and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be divided from
    * @return the new NDArray with the result of other / this
    */
   public NDArray rdiv(NDArray other) {
      return map(newZeroArray(),
                 other,
                 (v1, v2) -> v2 / v1);
   }

   /**
    * Divides each element's value from the given scalar (e.g. scalar - element)
    *
    * @param value the value to divide
    * @return the new NDArray with the scalar value divided
    */
   public NDArray rdiv(double value) {
      return mapScalar(newZeroArray(),
                       value,
                       (v1, v2) -> v2 / v1);
   }

   /**
    * Performs a column or row vector division dividing the values in this NDArray from the other NDArray to each row or
    * column in this NDArray as specified by the given axis parameter.
    *
    * @param other the other NDArray whose values will be divided
    * @param axis  the axis
    * @return the new NDArray with the result of this / other
    */
   public NDArray rdiv(NDArray other, Axis axis) {
      return mapVector(newZeroArray(),
                       other,
                       axis,
                       (v1, v2) -> v2 / v1);
   }

   /**
    * Divides the values in the this NDArray from the other NDArray in-place. Basic broadcasting will occur for scalar,
    * vector, and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be divided from
    * @return this NDArray with the result of other / this
    */
   public NDArray rdivi(NDArray other) {
      return map(this,
                 other,
                 (v1, v2) -> v2 / v1);
   }

   /**
    * Divides each element's value from the given scalar (e.g. scalar - element) in place
    *
    * @param value the value to divide
    * @return thisNDArray with the scalar value divided
    */
   public NDArray rdivi(double value) {
      return mapScalar(this,
                       value,
                       (v1, v2) -> v2 / v1);
   }

   /**
    * Performs a column or row vector division dividing the values in this NDArray from the other NDArray to each row or
    * column in this NDArray as specified by the given axis parameter in-place.
    *
    * @param other the other NDArray whose values will be divided
    * @param axis  the axis
    * @return this NDArray with the result of this / other
    */
   public NDArray rdivi(NDArray other, Axis axis) {
      return mapVector(this,
                       other,
                       axis,
                       (v1, v2) -> v2 / v1);
   }

   /**
    * Gets the number of rows in the NDArray
    *
    * @return the number of rows in the NDArray
    */
   public int numRows() {
      return dimension(Axis.ROW);
   }

   /**
    * Subtracts the values in the this NDArray from the other NDArray. Basic broadcasting will occur for scalar, vector,
    * and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be subtracted from
    * @return the new NDArray with the result of other - this
    */
   public NDArray rsub(NDArray other) {
      return map(newZeroArray(), other, (v1, v2) -> v2 - v1);
   }

   /**
    * Subtracts each element's value from the given scalar (e.g. scalar - element)
    *
    * @param value the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public NDArray rsub(double value) {
      return mapScalar(newZeroArray(), value, (v1, v2) -> v2 - v1);
   }

   /**
    * Performs a column or row vector subtraction subtracting the values in this NDArray from the other NDArray to each
    * row or column in this NDArray as specified by the given axis parameter.
    *
    * @param other the other NDArray whose values will be subtracted
    * @param axis  the axis
    * @return the new NDArray with the result of this - other
    */
   public NDArray rsub(NDArray other, Axis axis) {
      return mapVector(newZeroArray(), other, axis, (v1, v2) -> v2 - v1);
   }

   /**
    * Subtracts each element's value from the given scalar (e.g. scalar - element)  in-place.
    *
    * @param value the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public NDArray rsubi(double value) {
      return mapScalar(this, value, (v1, v2) -> v2 - v1);
   }

   /**
    * Subtracts the values in the this NDArray from the other NDArray in-place. Basic broadcasting will occur for
    * scalar, vector, and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be subtracted from
    * @return the new NDArray with the result of other - this
    */
   public NDArray rsubi(NDArray other) {
      return map(this, other, (v1, v2) -> v2 - v1);
   }

   /**
    * Performs a column or row vector subtraction subtracting the values in this NDArray from the other NDArray to each
    * row or column in this NDArray as specified by the given axis parameter in-place.
    *
    * @param other the other NDArray whose values will be subtracted
    * @param axis  the axis
    * @return the new NDArray with the result of this - other
    */
   public NDArray rsubi(NDArray other, Axis axis) {
      return mapVector(this, other, axis, (v1, v2) -> v2 - v1);
   }

   /**
    * Scalar value double.
    *
    * @return the double
    */
   public double scalarValue() {
      return get(0);
   }

   /**
    * Selects all values matching the given predicate.
    *
    * @param predicate the predicate to test
    * @return new NDArray with values passing the given predicate and zeros elsewhere
    */
   public NDArray select(DoublePredicate predicate) {
      return mapOperator(newZeroArray(),
                         v -> predicate.test(v) ? v : 0f);
   }

   /**
    * Selects all values in this NDArray whose corresponding element in the given predicate NDArray is not zero. Basic
    * broadcasting will occur for scalar, vector, and matrix NDArrays.
    *
    * @param predicate the predicate NDArray test
    * @return new NDArray with values passing the given predicate and zeros elsewhere
    */
   public NDArray select(NDArray predicate) {
      return map(newZeroArray(),
                 predicate,
                 (v1, v2) -> v2 != 0 ? v1 : 0f);
   }

   /**
    * Selects all values matching the given predicate in-place.
    *
    * @param predicate the predicate to test
    * @return this NDArray with values passing the given predicate and zeros elsewhere
    */
   public NDArray selecti(DoublePredicate predicate) {
      return mapOperator(this, v -> predicate.test(v) ? v : 0f);
   }

   /**
    * Selects all values in this NDArray whose corresponding element in the given predicate NDArray is not zero
    * in-place. Basic broadcasting will occur for scalar, vector, and matrix NDArrays.
    *
    * @param predicate the predicate NDArray test
    * @return this NDArray with values passing the given predicate and zeros elsewhere
    */
   public NDArray selecti(NDArray predicate) {
      return map(this,
                 predicate,
                 (v1, v2) -> v2 != 0 ? v1 : 0f);
   }

   /**
    * Sets the value at the given indices
    *
    * @param indices the indices
    * @param value   the value
    * @return This NDArray
    */
   public NDArray set(int[] indices, double value) {
      int[] dims = ensureCorrectIndices(indices);
      return set(dims[0], dims[1], dims[2], dims[3], value);
   }

   /**
    * Sets the value at the given indices
    *
    * @param row     the row
    * @param column  the column
    * @param kernel  the kernel
    * @param channel the channel
    * @param value   the value
    * @return This NDArray
    */
   public abstract NDArray set(int row, int column, int kernel, int channel, double value);

   /**
    * Sets the value at the given indices.
    *
    * @param row    the row
    * @param column the column
    * @param kernel the kernel
    * @param value  the value
    * @return This NDArray
    */
   public NDArray set(int row, int column, int kernel, double value) {
      return set(row, column, kernel, 0, value);
   }

   /**
    * Sets the value at the given indices.
    *
    * @param row    the row
    * @param column the column
    * @param value  the value
    * @return This NDArray
    */
   public NDArray set(int row, int column, double value) {
      return set(row, column, 0, 0, value);
   }

   /**
    * Sets the value at the given indices.
    *
    * @param row   the row
    * @param value the value
    * @return This NDArray
    */
   public NDArray set(int row, double value) {
      return set(row, 0, 0, 0, value);
   }

   /**
    * Replaces the slice at the given index.
    *
    * @param slice    the slice
    * @param newSlice the new slice to use at the given slice index
    */
   protected abstract void setSlice(int slice, NDArray newSlice);

   /**
    * Sets the vector at the given index along the given axis. Basic broadcasting will be preformed.
    *
    * @param index the index of the row / column to set
    * @param axis  the axis (row or column)
    * @param other the vector(s) to use to set.
    * @return this NDArray
    */
   public NDArray setVector(int index, Axis axis, NDArray other) {
      IntStream.range(0, numSlices).forEach(i -> {
         if (other.order == 1) {
            setVector(i, index, axis, other);
         } else {
            setVector(i, index, axis, other.tensorSlice(i));
         }
      });
      return this;
   }

   /**
    * Sets the vector at the given index along the given axis at the given slice index.
    *
    * @param sliceIndex the index of the slice whose vector will be set
    * @param index      the index of the row / column to set
    * @param axis       the axis (row or column)
    * @param vector     the vector to use to set.
    * @return this NDArray
    */
   public NDArray setVector(int sliceIndex, int index, Axis axis, NDArray vector) {
      checkArgument(axis.isRowOrColumn(), "Axis (" + axis + ") not supported.");
      checkArgument(vector.order < 2, "Order (" + vector.order + ") not supported as vector");
      NDArray slice = tensorSlice(sliceIndex);
      for (int i = 0; i < dimension(axis.T()); i++) {
         if (axis == Axis.ROW) {
            slice.set(index, i, vector.get(i));
         } else {
            slice.set(i, index, vector.get(i));
         }
      }
      return this;
   }

   /**
    * Returns the shape of the NDArray as an int array.
    *
    * @return the shape of the NDArray
    */
   public int[] shape() {
      return Arrays.copyOf(shape, shape.length);
   }

   /**
    * Gets the slice at the given kernel and channel. All changes made to the slice will be reflected in the NDArray.
    *
    * @param kernel  the kernel
    * @param channel the channel
    * @return The slice NDArray
    */
   public NDArray tensorSlice(int kernel, int channel) {
      return tensorSlice(toSliceIndex(kernel, channel));
   }

   /**
    * Gets the slice at the given index. All changes made to the slice will be reflected in the NDArray.
    *
    * @param index the index
    * @return The slice NDArray
    */
   public abstract NDArray tensorSlice(int index);

   /**
    * Applies the given binary function to the slices of this NDArray and the given other NDArray. If the given other
    * NDArray is of order <=2, it is applied to each slice, otherwise it should have the same number of slices. Whether
    * this NDArray is modified (via reuse of the * slices) or a new one created is dependent on the given function.
    *
    * @param other    the other NDArray
    * @param function the function to apply
    * @return the resulting NDArray
    */
   public NDArray sliceBinaryOperation(NDArray other, BiFunction<NDArray, NDArray, NDArray> function) {
      NDArray[] out = new NDArray[numSlices];
      if (other.order > 2) {
         checkArgument(other.sliceLength() == sliceLength(),
                       "Slice length mismatch (" + sliceLength() + ") != (" + other.sliceLength() + ")");
         sliceStream().forEach(t -> out[t.v1] = function.apply(t.v2, other.tensorSlice(t.v1)));
      } else {
         sliceStream().forEach(t -> out[t.v1] = function.apply(t.v2, other));
      }
      return getFactory().fromLayers(numKernels(), numChannels(), out);
   }

   /**
    * Calculates the dot product of vectors between this NDArray and the given NDArray on a per slice basis.
    *
    * @param other the other NDArray to calculate the dot product with
    * @return The sum of the dot products across the slices.
    */
   public NDArray sliceDot(NDArray other) {
      checkArgument(matrixLength == other.matrixLength,
                    "Length mismatch (" + matrixLength + ")  != (" + other.matrixLength);
      return sliceBinaryOperation(other, (a1, a2) -> getFactory().constant(a1.dot(a2), 1));
   }

   /**
    * Returns the length of a slice in this NDArray
    *
    * @return the length of a slice in this NDArray
    */
   public int sliceLength() {
      return matrixLength;
   }

   /**
    * Calculates the max value per slice
    *
    * @return NDArray of max value per slice
    */
   public NDArray sliceMax() {
      return sliceUnaryOperation(v -> getFactory().zeros(1)
                                                  .set(0, v.max()));
   }

   /**
    * Calculates the mean value per slice
    *
    * @return NDArray of mean value per slice
    */
   public NDArray sliceMean() {
      return sliceUnaryOperation(v -> getFactory().zeros(1)
                                                  .set(0, v.mean()));
   }

   /**
    * Calculates the min value per slice
    *
    * @return NDArray of min value per slice
    */
   public NDArray sliceMin() {
      return sliceUnaryOperation(v -> getFactory().zeros(1)
                                                  .set(0, v.min()));
   }

   /**
    * Calculates the norm1 per slice
    *
    * @return NDArray of norm1 per slice
    */
   public NDArray sliceNorm1() {
      return sliceUnaryOperation(v -> getFactory().zeros(1)
                                                  .set(0, v.norm1()));
   }

   /**
    * Calculates the norm2 per slice
    *
    * @return NDArray of norm2 per slice
    */
   public NDArray sliceNorm2() {
      return sliceUnaryOperation(v -> getFactory().zeros(1)
                                                  .set(0, v.norm2()));
   }

   /**
    * Generates a stream containing tuples of slice index and slice.
    *
    * @return the stream slice index and slice
    */
   public Stream<Tuple2<Integer, NDArray>> sliceStream() {
      return IntStream.range(0, slices()).mapToObj(i -> $(i, tensorSlice(i))).parallel();
   }

   /**
    * Calculates the sum per slice
    *
    * @return NDArray of sum per slice
    */
   public NDArray sliceSum() {
      return sliceUnaryOperation(v -> getFactory().zeros(1)
                                                  .set(0, v.sum()));
   }

   /**
    * Calculates the sum of squares per slice
    *
    * @return NDArray of sum of squares per slice
    */
   public NDArray sliceSumOfSquares() {
      return sliceUnaryOperation(v -> getFactory().zeros(1)
                                                  .set(0, v.sumOfSquares()));
   }

   /**
    * Applies the given unary function to the slices of this NDArray. Whether this NDArray is modified (via reuse of the
    * slices) or a new one created is dependent on the given function.
    *
    * @param function the function to apply
    * @return the resulting NDArray
    */
   public NDArray sliceUnaryOperation(Function<NDArray, NDArray> function) {
      NDArray[] out = new NDArray[numSlices];
      sliceStream().forEach(t -> out[t.v1] = function.apply(t.v2));
      return getFactory().fromLayers(numKernels(), numChannels(), out);
   }

   /**
    * Gets the number of slices in the NDArray
    *
    * @return the number of slices
    */
   public int slices() {
      return numSlices;
   }

   /**
    * Iterator over the non-zero entries in the NDArray
    *
    * @return the non-zero entry iterator
    */
   public Iterator<Entry> sparseIterator() {
      return Iterators.filter(iterator(), e -> e.getValue() != 0f);
   }

   public Iterator<Entry> sparseOrderedIterator() {
      return sparseIterator();
   }


   public NDArray toUnitVector() {
      checkArgument(isVector(), "NDArray must be a vector");
      float mag = norm2();
      return div(mag);
   }


   /**
    * Subtracts the values in the other NDArray to this one. Basic broadcasting will occur for scalar, vector, and
    * matrix NDArrays.
    *
    * @param other the other NDArray whose values will be subtracted
    * @return the new NDArray with the result of this - other
    */
   public NDArray sub(NDArray other) {
      return map(newZeroArray(),
                 other,
                 Operator::subtract);
   }

   /**
    * Performs a column or row vector subtraction subtracting the values in the other NDArray to each row or column in
    * this NDArray as specified by the given axis parameter.
    *
    * @param other the other NDArray whose values will be subtracted
    * @param axis  the axis
    * @return the new NDArray with the result of this - other
    */
   public NDArray sub(NDArray other, Axis axis) {
      return mapVector(newZeroArray(),
                       other,
                       axis,
                       Operator::subtract);
   }

   /**
    * Subtracts a scalar value to each element in the NDArray
    *
    * @param value the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public NDArray sub(double value) {
      return mapScalar(newZeroArray(),
                       value,
                       Operator::subtract);
   }

   /**
    * Subtracts a scalar value to each element in the NDArray in-place
    *
    * @param value the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public NDArray subi(double value) {
      return mapScalar(this,
                       value,
                       Operator::subtract);
   }

   /**
    * Subtracts the values in the other NDArray to this one  in-place. Basic broadcasting will occur for scalar, vector,
    * and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be subtracted
    * @return the new NDArray with the result of this - other
    */
   public NDArray subi(NDArray other) {
      return map(this,
                 other,
                 Operator::subtract);
   }

   /**
    * Performs a column or row vector subtraction subtracting the values in the other NDArray to each row or column in
    * this NDArray as specified by the given axis parameter  in-place.
    *
    * @param other the other NDArray whose values will be subtracted
    * @param axis  the axis
    * @return the new NDArray with the result of this - other
    */
   public NDArray subi(NDArray other, Axis axis) {
      return mapVector(this,
                       other,
                       axis,
                       Operator::subtract);
   }

   /**
    * Calculates the sum of all values in the NDArray
    *
    * @return the sum all values
    */
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

   /**
    * Calculates the sum along each axis and per slice
    *
    * @param axis The axis to calculate the sum for
    * @return An NDArray of the sum
    */
   public NDArray sum(Axis axis) {
      Validation.checkArgument(axis.isRowOrColumn(),
                               "Axis (" + axis + ") is not supported");
      int[] newShape = shape();
      newShape[axis.T().ordinal] = 1;
      NDArray out = getFactory().zeros(newShape);

      sliceStream().forEach(t -> {
         NDArray outSlice = out.tensorSlice(t.v1);
         t.v2.iterator().forEachRemaining(e -> {
            int i = axis.select(e.row, e.column);
            outSlice.set(i, outSlice.get(i) + e.getValue());
         });
      });

      return out;
   }

   /**
    * Calculates the sum of squares of all values in the NDArray
    *
    * @return the sum of squares
    */
   public float sumOfSquares() {
      return (float) sliceStream().mapToDouble(t ->
                                                  Streams.asStream(t.v2)
                                                         .mapToDouble(e -> Math.pow(e.getValue(), 2))
                                                         .sum()
                                              ).sum();
   }

   /**
    * Tests the given predicate on the values in the NDArray returning 1 when TRUE and 0 when FALSE
    *
    * @param predicate the predicate to test
    * @return new NDArray with test results
    */
   public NDArray test(DoublePredicate predicate) {
      return mapOperator(newZeroArray(), v -> predicate.test(v) ? 1 : 0f);
   }

   /**
    * Tests the given predicate on the values in the NDArray returning 1 when TRUE and 0 when FALSE in-place
    *
    * @param predicate the predicate to test
    * @return this with test results
    */
   public NDArray testi(DoublePredicate predicate) {
      return mapOperator(this, v -> predicate.test(v) ? 1 : 0f);
   }

   /**
    * Generates a JBlas DoubleMatrix view of the data
    *
    * @return the double matrix
    */
   public abstract DoubleMatrix toDoubleMatrix();

   /**
    * Generates a float view of the NDArray
    *
    * @return 1d array of float values
    */
   public float[] toFloatArray() {
      float[] out = new float[(int) length()];
      sparseIterator().forEachRemaining(e -> out[(int) e.getIndex()] = e.getValue());
      return out;
   }

   public double[] toDoubleArray() {
      double[] out = new double[(int) length()];
      sparseIterator().forEachRemaining(e -> out[(int) e.getIndex()] = e.getValue());
      return out;
   }

   /**
    * Generates a JBlas FloatMatrix view of the data
    *
    * @return the float matrix
    */
   public abstract FloatMatrix toFloatMatrix();

   /**
    * To index long.
    *
    * @param indices the indices
    * @return the long
    */
   protected long toIndex(int[] indices) {
      return toLongIndex(indices, shape);
   }

   @Override
   public JsonEntry toJson() {
      JsonEntry ndarray = JsonEntry.object()
                                   .addProperty("shape", shape())
                                   .addProperty("dense", isDense());
      JsonEntry array = JsonEntry.array();
      for (int i = 0; i < slices(); i++) {
         array.addValue(tensorSlice(i).toFloatArray());
      }
      ndarray.addProperty("data", array);
      return ndarray;
   }

   private class IndicesIterator implements Iterator<int[]> {
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

   /**
    * Defines an entry, or cell, in the NDArray with corresponding indices and value
    */
   public class Entry {
      /**
       * The Row.
       */
      final int row, /**
       * The Column.
       */
      column, /**
       * The Kernel.
       */
      kernel, /**
       * The Channel.
       */
      channel;

      /**
       * Instantiates a new Entry.
       *
       * @param indices the indices
       */
      protected Entry(int[] indices) {
         this(indices[0], indices[1], indices[2], indices[3]);
      }

      /**
       * Instantiates a new Entry.
       *
       * @param row     the row
       * @param column  the column
       * @param kernel  the kernel
       * @param channel the channel
       */
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

      /**
       * Gets channel.
       *
       * @return the channel
       */
      public int getChannel() {
         return channel;
      }

      /**
       * Gets column.
       *
       * @return the column
       */
      public int getColumn() {
         return column;
      }

      /**
       * Gets index.
       *
       * @return the index
       */
      public long getIndex() {
         return toIndex(new int[]{row, column, kernel, channel});
      }

      /**
       * Gets index.
       *
       * @param axis the axis
       * @return the index
       */
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

      /**
       * Get indicies int [ ].
       *
       * @return the int [ ]
       */
      public int[] getIndicies() {
         return new int[]{
            getRow(),
            getColumn(),
            getKernel(),
            getChannel()
         };
      }

      /**
       * Gets kernel.
       *
       * @return the kernel
       */
      public int getKernel() {
         return kernel;
      }

      /**
       * Gets row.
       *
       * @return the row
       */
      public int getRow() {
         return row;
      }

      /**
       * Gets value.
       *
       * @return the value
       */
      public float getValue() {
         return get(row, column, kernel, channel);
      }

      /**
       * Sets value.
       *
       * @param value the value
       */
      public void setValue(double value) {
         set(row, column, kernel, channel, value);
      }

      @Override
      public int hashCode() {
         return Objects.hash(row, column, kernel, channel);
      }

      /**
       * Matrix index int.
       *
       * @return the int
       */
      public int matrixIndex() {
         return toIndex(row, shape[0], column, shape[1]);
      }

      /**
       * Slice index int.
       *
       * @return the int
       */
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


   private String toString(FloatMatrix matrix) {
      return matrix.toString("%f", "[", "]", ", ", "],\n  [");
   }

   @Override
   public String toString() {
      StringBuilder builder = new StringBuilder("[[")
                                 .append(toString(tensorSlice(0).toFloatMatrix()));
      for (int i = 1; i < Math.min(slices(), 10); i++) {
         builder.append("]").append(System.lineSeparator())
                .append(" [")
                .append(toString(tensorSlice(i).toFloatMatrix()));
      }

      if (slices() > 10) {
         if (slices() > 11) {
            builder.append("]")
                   .append(System.lineSeparator())
                   .append("  ...")
                   .append(System.lineSeparator());
         }
         builder.append(" [").append(toString(tensorSlice(10).toFloatMatrix()));
         for (int i = Math.max(11, slices() - 10);
              i < Math.max(Math.min(20, slices()), slices());
              i++) {
            builder.append("]").append(System.lineSeparator())
                   .append(" [")
                   .append(toString(tensorSlice(i).toFloatMatrix()));
         }
      }

      return builder.append("]]").toString();
   }

   public abstract NDArray slice(int from, int to);

   public abstract NDArray slice(int iFrom, int iTo, int jFrom, int jTo);

   public abstract NDArray slice(@NonNull Axis axis, int... indexes);
}//END OF NDArray
