package com.gengoai.apollo.linear;

import com.gengoai.Copyable;
import com.gengoai.collection.Iterators;
import com.gengoai.collection.Streams;
import com.gengoai.conversion.Cast;
import com.gengoai.json.JsonEntry;
import com.gengoai.json.JsonSerializable;
import com.gengoai.math.Math2;
import com.gengoai.math.NumericComparison;
import com.gengoai.math.Operator;
import com.gengoai.math.Optimum;
import com.gengoai.string.StringUtils;
import com.gengoai.tuple.Tuple2;
import org.apache.commons.math3.util.FastMath;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;
import java.util.function.*;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import static com.gengoai.Validation.*;
import static com.gengoai.apollo.linear.NDArrayFactory.DENSE;
import static com.gengoai.collection.Iterators.zipWithIndex;
import static com.gengoai.tuple.Tuples.$;

/**
 * An n-dimension array of float values used for vectors, matrices, and tensors.
 *
 * @author David B. Bracewell
 */
public abstract class NDArray implements Copyable<NDArray>, Serializable, JsonSerializable, Iterable<NDArray.Entry> {
   private static final long serialVersionUID = 1L;
   private static NumberFormat decimalFormatter = new DecimalFormat(" 0.000000;-0");
   private final int[] shape;
   private final long length;
   private final int matrixLength;
   private final int numSlices;
   private int order;
   private Object label;
   private Object predicted;
   private float weight;

   /**
    * Default Constructor
    *
    * @param shape The shape of the new NDArray
    * @throws IllegalArgumentException if the length of the shape array is greater than four.
    */
   protected NDArray(int[] shape) {
      checkArgument(shape.length <= 4, () -> invalidNumberOfIndices(shape.length));
      this.shape = new int[]{1, 1, 1, 1};
      System.arraycopy(shape, 0, this.shape, 0, shape.length);
      this.order = orderOf(this.shape);
      this.matrixLength = this.shape[0] * this.shape[1];
      this.numSlices = this.shape[2] * this.shape[3];
      this.length = this.matrixLength * this.numSlices;
   }

   private static void addLabelProperty(JsonEntry entry, String propertyName, Object label) {
      if (label instanceof String ||
             label instanceof Number ||
             label instanceof Boolean) {
         entry.addProperty(propertyName, label);
      } else {
         entry.addProperty(propertyName, JsonEntry.object()
                                                  .addProperty("class", label.getClass())
                                                  .addProperty("object", label));
      }
   }

   /**
    * Axis not supported string.
    *
    * @param axis the axis
    * @return the string
    */
   static String axisNotSupported(Axis axis) {
      return "Axis (" + axis + ") is not supported.";
   }

   /**
    * Method for constructing an NDArray from JsonEntry for use in serializing / deserializing to/from json.
    *
    * @param entry The <code>JsonEntry</code> to parse the NDArray from
    * @return The NDArray from the JsonEntry
    */
   public static NDArray fromJson(JsonEntry entry) {
      NDArrayFactory factory = entry.getValProperty("factory").as(NDArrayFactory.class);
      NDArray ndArray = factory.zeros(entry.getValProperty("shape").asIntegerValueArray());
      if (entry.hasProperty("weight")) {
         ndArray.setWeight(entry.getFloatProperty("weight"));
      }
      if (entry.hasProperty("label")) {
         ndArray.setLabel(getLabelProperty(entry.getProperty("label")));
      }
      if (entry.hasProperty("predicted")) {
         ndArray.setPredicted(getLabelProperty(entry.getProperty("predicted")));
      }
      JsonEntry array = entry.getProperty("data");
      if (ndArray.isDense()) {
         zipWithIndex(array.elementIterator())
            .forEachRemaining(e -> zipWithIndex(e.getKey().elementIterator())
                                      .forEachRemaining(
                                         v -> ndArray.setIndexedValue(e.getValue(),
                                                                      v.getValue(),
                                                                      v.getKey().getAsFloat())));
      } else {
         zipWithIndex(array.elementIterator())
            .forEachRemaining(e -> e.getKey().propertyIterator()
                                    .forEachRemaining(v -> ndArray.setIndexedValue(e.getValue(),
                                                                                   Integer.parseInt(v.getKey()),
                                                                                   v.getValue().getAsFloat())));
      }
      return ndArray;
   }

   private static Object getLabelProperty(JsonEntry entry) {
      if (entry.isString()) {
         return entry.getAsString();
      } else if (entry.isNumber()) {
         return entry.getAsFloat();
      } else if (entry.isBoolean()) {
         return entry.getAsBoolean();
      }
      return entry.getValProperty("object").as(entry.getValProperty("class").asClass());
   }

   /**
    * Invalid number of indices string.
    *
    * @param length the length
    * @return the string
    */
   static String invalidNumberOfIndices(int length) {
      return "Invalid number of indices (" + length + ")";
   }

   /**
    * Length mismatch string.
    *
    * @param l1 the l 1
    * @param l2 the l 2
    * @return the string
    */
   static String lengthMismatch(int l1, int l2) {
      return "Length mismatch (" + l1 + ") != (" + l2 + ")";
   }

   /**
    * Order not supported string.
    *
    * @param order the order
    * @return the string
    */
   static String orderNotSupported(int order) {
      return "Order (" + order + ") is not supported.";
   }

   protected static int orderOf(int[] shape) {
      if (shape[Axis.CHANNEL.ordinal] > 1) {
         return 4;
      } else if (shape[Axis.KERNEL.ordinal] > 1) {
         return 3;
      } else if (shape[Axis.ROW.ordinal] > 1 && shape[Axis.COLUMN.ordinal] > 1) {
         return 2;
      } else if (shape[Axis.ROW.ordinal] > 1 || shape[Axis.COLUMN.ordinal] > 1) {
         return 1;
      }
      return 0;
   }

   /**
    * Slice out bounds string.
    *
    * @param slice     the slice
    * @param numSlices the num slices
    * @return the string
    */
   static String sliceOutBounds(int slice, int numSlices) {
      return "Slice index (" + slice + ") out of bounds [0, " + numSlices + ")";
   }

   /**
    * Calculates a dimension-2 major index (e.g. column major) given two axis indices and their dimensions
    *
    * @param ax1    the index along axis 1
    * @param dimAx1 the dimension of axis 1
    * @param ax2    the index along axis 2
    * @return the encoded index
    */
   public static int toIndex(int ax1, int dimAx1, int ax2) {
      return ax1 + (dimAx1 * ax2);
   }

   /**
    * Flips the matrix on its diagonal switching the rows and columns. (This is done per slice)
    *
    * @return the transposed array
    */
   public NDArray T() {
      return sliceUnaryOperation(v -> {
         NDArray out = getFactory().zeros(v.numCols(), v.numRows());
         v.forEachSparse(e -> out.set(e.getColumn(), e.getRow(), e.getValue()));
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
                       Operator::add,
                       false);
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
                 Operator::add,
                 false);
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
                       Operator::add,
                       false);
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
                       Operator::add,
                       false);
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
                 Operator::add,
                 false);
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
                       Operator::add,
                       false);
   }

   /**
    * Adjusts the value at the give slice and matrix index by adding the given value.
    *
    * @param sliceIndex  the slice index
    * @param matrixIndex the matrix index
    * @param value       the value to add
    * @return this NDArray
    */
   public abstract NDArray adjustIndexedValue(int sliceIndex, int matrixIndex, double value);

   /**
    * Adjusts the value of the first slice at the given matrix index by adding the given value.
    *
    * @param matrixIndex the matrix index
    * @param value       the value to add
    * @return this NDArray
    */
   public NDArray adjustIndexedValue(int matrixIndex, double value) {
      return adjustIndexedValue(0, matrixIndex, value);
   }

   /**
    * Calculates the index of the maximum along each row or column based on the given axis.
    *
    * @param axis the axis (row/column) to calculate the max for
    * @return array of int array of row/column indexes relating to max values per slice
    */
   public NDArray argMax(Axis axis) {
      return optimumPosition(axis,
                             Optimum.MAXIMUM);
   }

   /**
    * Calculates the index of the minimum along each row or column based on the given axis.
    *
    * @param axis the axis (row/column) to calculate the minimum for
    * @return array of int array of row/column indexes relating to minimum values per slice
    */
   public NDArray argMin(Axis axis) {
      return optimumPosition(axis,
                             Optimum.MINIMUM);
   }

   private NDArray asNDArray(Object o, int dimension) {
      if (o == null) {
         return getFactory().empty();
      } else if (o instanceof Number) {
         return getFactory().zeros(dimension)
                            .set(Cast.<Number>as(label).intValue(), 1f);
      }
      NDArray nd = Cast.as(o, NDArray.class);
      notNull(nd, "Cannot create NDArray from object.");
      checkState(nd.length == dimension, lengthMismatch(dimension, (int) nd.length));
      return nd;
   }

   /**
    * Column iterator iterator.
    *
    * @param column the column
    * @return the iterator
    */
   public Iterator<Entry> columnIterator(final int column) {
      return new Iterator<Entry>() {
         int slice = 0;
         int row = 0;

         @Override
         public boolean hasNext() {
            return row < numRows() && slice < numSlices;
         }

         @Override
         public Entry next() {
            checkElementIndex(row, numRows());
            Entry e = new Entry(slice, toMatrixIndex(row, column));
            row++;
            if (row > numRows()) {
               row = 0;
               slice++;
            }
            return e;
         }
      };
   }

   /**
    * Compresses the memory used by the NDArray. (Only useful for sparse implementations)
    *
    * @return this NDArray
    */
   public NDArray compress() {
      return this;
   }

   @Override
   public NDArray copy() {
      return copyData().setLabel(label)
                       .setWeight(weight)
                       .setPredicted(predicted);
   }

   /**
    * Copies the raw values from this NDArray
    *
    * @return An NDArray with the same values
    */
   protected abstract NDArray copyData();

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
      checkArgument(indices.length > 0 && indices.length <= 4, () -> invalidNumberOfIndices(shape.length));
      switch (indices.length) {
         case 1:
            return decrement(indices[0], value);
         case 2:
            return decrement(indices[0], indices[1], value);
         case 3:
            return decrement(indices[0], indices[1], indices[2], value);
         default:
            return decrement(indices[0], indices[1], indices[2], indices[3], value);
      }
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
      return adjustIndexedValue(toSliceIndex(kernel, channel), toMatrixIndex(row, column), -value);
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
      return adjustIndexedValue(kernel, toMatrixIndex(row, column), -value);
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
      return adjustIndexedValue(toMatrixIndex(row, column), -value);
   }

   /**
    * Decrements the value of the NDArray at the given indices.
    *
    * @param row   the row
    * @param value the value to decrement by
    * @return This NDArray
    */
   public NDArray decrement(int row, double value) {
      return adjustIndexedValue(row, -value);
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
      return map(newZeroArray(),
                 other,
                 Operator::divide,
                 true);
   }

   /**
    * Divides a scalar value to each element in the NDArray
    *
    * @param value the value to divide
    * @return the new NDArray with the scalar value divided
    */
   public NDArray div(double value) {
      return mapScalar(newZeroArray(),
                       value,
                       Operator::divide,
                       true);
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
      return mapVector(newZeroArray(),
                       other,
                       axis,
                       Operator::divide,
                       true);
   }

   /**
    * Divides a scalar value to each element in the NDArray in-place.
    *
    * @param value the value to divide
    * @return this NDArray with the scalar value divided
    */
   public NDArray divi(double value) {
      return mapScalar(this,
                       value,
                       Operator::divide,
                       true);
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
      return map(this,
                 other,
                 Operator::divide,
                 true);
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
      return mapVector(this,
                       other,
                       axis,
                       Operator::divide,
                       true);
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
      checkArgument(axis.isRowOrColumn(), () -> axisNotSupported(axis));
      checkArgument(dimension(axis.T()) == other.dimension(axis.T()), () -> lengthMismatch(dimension(axis.T()),
                                                                                           other.dimension(axis.T())));
      NDArray out = getFactory().zeros(axis.T().set(shape(), 1));
      forEachSlice((si, slice) -> {
         NDArray os = other.getSlice(si);
         slice.forEachSparse(e -> out.adjustIndexedValue(si,
                                                         e.matrixIndex,
                                                         os.get(e.matrixIndex) * e.getValue()));
      });
      return out;
   }

   /**
    * Dot nd array.
    *
    * @param other the other
    * @return the nd array
    */
   public NDArray dot(NDArray other) {
      checkArgument(matrixLength == other.matrixLength, () -> lengthMismatch(matrixLength, other.matrixLength));
      return sliceBinaryOperation(other, (n, os) -> {
         double dot = 0d;
         NDArray small = n.size() > os.size() ? os : n;
         NDArray big = n.size() > os.size() ? n : os;
         for (Iterator<Entry> itr = small.sparseIterator(); itr.hasNext(); ) {
            Entry e = itr.next();
            dot += e.getValue() * big.getIndexedValue(0, e.matrixIndex);
         }
         return getFactory().scalar(dot);
      });
   }

   @Override
   public boolean equals(Object o) {
      if (o instanceof NDArray) {
         NDArray oa = Cast.as(o);
         if (Objects.equals(label, oa.label) &&
                Objects.equals(predicted, oa.predicted) &&
                Objects.equals(weight, oa.weight) &&
                Arrays.equals(shape, oa.shape)) {
            for (int i = 0; i < numSlices(); i++) {
               if (!Arrays.equals(getSlice(i).toFloatArray(), oa.getSlice(i).toFloatArray())) {
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
      forEach(e -> e.setValue(value));
      return this;
   }

   /**
    * Fills the NDArray with values generated from the given double supplier
    *
    * @param supplier the supplier to use to set all cells in the NDArray
    * @return This NDArray
    */
   public NDArray fill(DoubleSupplier supplier) {
      forEach(e -> e.setValue(supplier.getAsDouble()));
      return this;
   }

   /**
    * Processes each slice of the NDArray using the given NDArray consumer
    *
    * @param sliceConsumer the slice consumer
    */
   public void forEachSlice(BiConsumer<Integer, NDArray> sliceConsumer) {
      for (int i = 0; i < numSlices(); i++) {
         sliceConsumer.accept(i, getSlice(i));
      }
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
   public float get(int... indices) {
      checkArgument(indices.length > 0 && indices.length <= 4, () -> invalidNumberOfIndices(indices.length));
      switch (indices.length) {
         case 1:
            return getIndexedValue(0, indices[0]);
         case 2:
            return getIndexedValue(0, toMatrixIndex(indices[0], indices[1]));
         case 3:
            return getIndexedValue(indices[2], toMatrixIndex(indices[0], indices[1]));
         default:
            return getIndexedValue(toSliceIndex(indices[2], indices[3]), toMatrixIndex(indices[0], indices[1]));
      }
   }

   /**
    * Gets the factory used to create NDArrays of this type
    *
    * @return the NDArrayFactory
    */
   public abstract NDArrayFactory getFactory();

   /**
    * Raw access to the value given at the slice and matrix index.
    *
    * @param sliceIndex  the index of the slice
    * @param matrixIndex the index in the matrix
    * @return the value at the given slice and matrix index
    */
   public abstract float getIndexedValue(int sliceIndex, int matrixIndex);

   /**
    * Gets the value at the given matrix index for the first slice
    *
    * @param matrixIndex the matrix index
    * @return the indexed value at the given matrix index
    */
   public float getIndexedValue(int matrixIndex) {
      return getIndexedValue(0, matrixIndex);
   }

   /**
    * Gets the label associated with the NDArray
    *
    * @param <T> the type of the label
    * @return the label
    */
   public <T> T getLabel() {
      return Cast.as(label);
   }

   /**
    * Sets the label associated with the NDArray
    *
    * @param label the label
    * @return This NDArray
    */
   public NDArray setLabel(Object label) {
      this.label = label;
      return this;
   }

   /**
    * Gets the label associated with the NDArray as a double value.
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
    * Gets the label associated with the NDArray as an NDArray
    *
    * @return the label as NDArray
    */
   public NDArray getLabelAsNDArray() {
      return getLabelAsNDArray(1);
   }

   /**
    * Gets the label associated with this NDArray as an NDArray (vector) with desired dimension.
    *
    * @param dimension the dimension
    * @return the label as nd array
    */
   public NDArray getLabelAsNDArray(int dimension) {
      return asNDArray(label, dimension);
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
      return asNDArray(predicted, 1);
   }

   /**
    * Gets label as nd array.
    *
    * @param dimension the dimension
    * @return the label as nd array
    */
   public NDArray getPredictedAsNDArray(int dimension) {
      return asNDArray(predicted, dimension);
   }

   /**
    * Gets the slice at the given kernel and channel  or this NDArray if it has order <= 2.. All changes made to the
    * slice will be reflected in the NDArray.
    *
    * @param kernel  the kernel
    * @param channel the channel
    * @return The slice NDArray
    */
   public NDArray getSlice(int kernel, int channel) {
      return getSlice(toSliceIndex(kernel, channel));
   }

   /**
    * Gets the slice at the given index or this NDArray if it has order <= 2. All changes made to the slice will be
    * reflected in the NDArray.
    *
    * @param index the index
    * @return The slice NDArray
    */
   public abstract NDArray getSlice(int index);

   /**
    * Gets the vector(s) at the given index along the given axis. This works across slices. In addition this will
    * broadcast row and column vectors, e.g. if the dimension along the given axis is 1 and a copy of the vector will
    * always be returned.
    *
    * @param index the index of the vector to retrieve
    * @param axis  the axis of the vector
    * @return the NDArray of vectors
    */
   public NDArray getVector(int index, Axis axis) {
      checkArgument(axis.isRowOrColumn(), () -> axisNotSupported(axis));
      if (dimension(axis) == 1) {
         return copyData();
      }
      return sliceUnaryOperation(slice -> {
         NDArray out = getFactory().zeros(axis.set(new int[]{slice.numRows(), slice.numCols()}, 1));
         for (Iterator<Entry> itr = sparseVectorIterator(index, axis); itr.hasNext(); ) {
            Entry e = itr.next();
            out.set(axis.T().select(e.getRow(), e.getColumn()), e.getValue());
         }
         return out;
      });
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
      checkArgument(indices.length > 0 && indices.length <= 4, () -> invalidNumberOfIndices(indices.length));
      switch (indices.length) {
         case 1:
            return increment(indices[0], value);
         case 2:
            return increment(indices[0], indices[1], value);
         case 3:
            return increment(indices[0], indices[1], indices[2], value);
         default:
            return increment(indices[0], indices[1], indices[2], indices[3], value);
      }
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
      return adjustIndexedValue(toSliceIndex(kernel, channel), toMatrixIndex(row, column), value);
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
      return adjustIndexedValue(kernel, toMatrixIndex(row, column), value);
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
      return adjustIndexedValue(0, toMatrixIndex(row, column), value);
   }

   /**
    * Increments the value of the NDArray at the given indices.
    *
    * @param row   the row
    * @param value the value to increment by
    * @return This NDArray
    */
   public NDArray increment(int row, double value) {
      return adjustIndexedValue(0, row, value);
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
    * Is column vector slices boolean.
    *
    * @return the boolean
    */
   public boolean isColumnVectorSlices() {
      return dimension(Axis.ROW) > 1 && dimension(Axis.COLUMN) == 1;
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
    * Is row vector slices boolean.
    *
    * @return the boolean
    */
   public boolean isRowVectorSlices() {
      return dimension(Axis.ROW) == 1 && dimension(Axis.COLUMN) > 1;
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
    * Is scalar slices boolean.
    *
    * @return the boolean
    */
   public boolean isScalarSlices() {
      return dimension(Axis.ROW) == 1 && dimension(Axis.COLUMN) == 1;
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
    * Is square slices boolean.
    *
    * @return the boolean
    */
   public boolean isSquareSlices() {
      return numRows() == numCols();
   }

   /**
    * Checks if the NDArray is a vector (row or column)
    *
    * @return True if the NDArray is a vector
    */
   public boolean isVector() {
      return isRowVector() || isColumnVector();
   }

   /**
    * Is vector slices boolean.
    *
    * @return the boolean
    */
   public boolean isVectorSlices() {
      return isColumnVectorSlices() || isRowVectorSlices();
   }

   @Override
   public Iterator<Entry> iterator() {
      return new IndexIterator();
   }

   /**
    * Iterator iterator.
    *
    * @param sparse the sparse
    * @return the iterator
    */
   public Iterator<Entry> iterator(boolean sparse) {
      return sparse ? sparseIterator() : iterator();
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
      return mapOperator(newZeroArray(),
                         operator,
                         false);
   }

   private NDArray map(NDArray out, NDArray other, DoubleBinaryOperator operator, boolean sparse) {
      switch (other.order) {
         case 0:
            return mapScalar(out,
                             other.get(0),
                             operator,
                             sparse);
         case 1:
            if (order == 1) {
               return mapTensor(out,
                                other,
                                operator,
                                sparse);
            }
            return mapVector(out,
                             other,
                             other.isRowVector() ? Axis.COLUMN : Axis.ROW,
                             operator,
                             sparse);
         default:
            return mapTensor(out,
                             other,
                             operator,
                             sparse);
      }
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
      return map(newZeroArray(),
                 other,
                 operator,
                 false);
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
                       operator,
                       false);
   }

   private NDArray mapOperator(NDArray out, DoubleUnaryOperator operator, boolean sparse) {
      forEachSlice((index, slice) -> {
         for (Iterator<Entry> itr = iterator(sparse); itr.hasNext(); ) {
            Entry e = itr.next();
            out.setIndexedValue(index,
                                e.matrixIndex,
                                operator.applyAsDouble(e.getValue()));
         }
      });
      return out;
   }

   private NDArray mapScalar(NDArray out, double scalar, DoubleBinaryOperator operator, boolean sparse) {
      forEachSlice((index, slice) -> {
         for (Iterator<Entry> itr = iterator(sparse); itr.hasNext(); ) {
            Entry e = itr.next();
            out.setIndexedValue(index,
                                e.matrixIndex,
                                operator.applyAsDouble(e.getValue(), scalar));
         }
      });
      return out;
   }

   /**
    * Performs the given operator on each sparse entry in the NDArray storing the result in a new NDArray
    *
    * @param operator the operation to perform
    * @return The new  NDArray with the operator applied to this NDArray's values
    */
   public NDArray mapSparse(DoubleUnaryOperator operator) {
      return mapOperator(newZeroArray(),
                         operator,
                         true);
   }

   /**
    * Performs the given operator on each sparse entry in the NDArray storing the result in a new NDArray
    *
    * @param other    the other
    * @param operator the operation to perform
    * @return The new  NDArray with the operator applied to this NDArray's values
    */
   public NDArray mapSparse(NDArray other, DoubleBinaryOperator operator) {
      return map(newZeroArray(),
                 other,
                 operator,
                 true);
   }

   private NDArray mapTensor(NDArray out, NDArray other, DoubleBinaryOperator operator, boolean sparse) {
      forEachSlice((index, slice) -> {
         NDArray oSlice = other.getSlice(index);
         for (Iterator<Entry> itr = iterator(sparse); itr.hasNext(); ) {
            Entry e = itr.next();
            out.setIndexedValue(e.sliceIndex,
                                e.matrixIndex,
                                operator.applyAsDouble(e.getValue(), oSlice.get(e.matrixIndex)));
         }
      });
      return out;
   }

   private NDArray mapVector(NDArray out, NDArray rowVector, Axis axis, DoubleBinaryOperator operator, boolean sparse) {
      forEachSlice((index, slice) -> {
         for (Iterator<Entry> itr = iterator(sparse); itr.hasNext(); ) {
            Entry e = itr.next();
            out.setIndexedValue(index,
                                e.matrixIndex,
                                operator.applyAsDouble(e.getValue(),
                                                       rowVector.get(axis.T().select(e.getRow(), e.getColumn()))));
         }
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
      return map(this,
                 other,
                 operator,
                 false);
   }

   /**
    * Applies an operation to the elements in this NDArray and given other NDArray using the given operator in-place.
    *
    * @param operator the operator to apply
    * @return this NDArray
    */
   public NDArray mapi(DoubleUnaryOperator operator) {
      return mapOperator(this,
                         operator,
                         false);
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
                       operator,
                       false);
   }

   /**
    * Applies an operation to the non-zero elements in this NDArray and given other NDArray using the given operator
    * in-place.
    *
    * @param operator the operator to apply
    * @return this NDArray
    */
   public NDArray mapiSparse(DoubleUnaryOperator operator) {
      return mapOperator(this,
                         operator,
                         true);
   }

   /**
    * Performs the given operator on each sparse entry in the NDArray storing the result in a new NDArray
    *
    * @param other    the other
    * @param operator the operation to perform
    * @return The new  NDArray with the operator applied to this NDArray's values
    */
   public NDArray mapiSparse(NDArray other, DoubleBinaryOperator operator) {
      return map(this,
                 other,
                 operator,
                 true);
   }

   private String matrixToString(int maxR, int maxC) {
      StringBuilder builder = new StringBuilder("[");
      builder.append(rowToString(0, maxC));
      int half = maxR / 2;
      boolean firstHalf = true;
      for (int i = 1; i < numRows(); i++) {
         builder.append(",");
         if (i > half && firstHalf) {
            firstHalf = false;
            int ni = Math.max(numRows() - half, i + 1);
            if (ni > i + 1) {
               builder.append(System.lineSeparator()).append("     ...").append(System.lineSeparator());
            }
            i = ni;
         }
         builder.append(System.lineSeparator()).append("  ").append(rowToString(i, maxC));
      }
      return builder.append("]").toString();
   }

   public NDArray max() {
      return sliceUnaryOperation(n -> getFactory().scalar(n.scalarMax()));
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
   public NDArray mean() {
      return sum().divi(matrixLength);
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

   public NDArray min() {
      return sliceUnaryOperation(n -> getFactory().scalar(n.scalarMin()));
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

   /**
    * Calculates the product of this and the given NDArray (i.e. matrix multiplication). Works on a per slice basis
    *
    * @param other The other NDArray to multiple
    * @return a new NDArray that is the result of this X other
    */
   public abstract NDArray mmul(NDArray other);

   /**
    * Multiplies the values in the other NDArray to this one element by element. Basic broadcasting will occur for
    * scalar, vector, and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be multiplied
    * @return the new NDArray with the result of this * other
    */
   public NDArray mul(NDArray other) {
      return map(newZeroArray(),
                 other,
                 Operator::multiply,
                 true);
   }

   /**
    * Multiplies a scalar value to each element in the NDArray
    *
    * @param value the value to multiplied
    * @return the new NDArray with the scalar value multiplied
    */
   public NDArray mul(double value) {
      return mapScalar(newZeroArray(),
                       value,
                       Operator::multiply,
                       true);
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
      return mapVector(newZeroArray(),
                       other,
                       axis,
                       Operator::multiply, true);
   }

   /**
    * Multiplies a scalar value to each element in the NDArray in-place.
    *
    * @param value the value to multiplied
    * @return this NDArray with the scalar value multiplied
    */
   public NDArray muli(double value) {
      return mapScalar(this,
                       value,
                       Operator::multiply,
                       true);
   }

   /**
    * Multiplies the values in the other NDArray to this one element by element in-place. Basic broadcasting will occur
    * for scalar, vector, and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be multiplied
    * @return this NDArray with the result of this * other
    */
   public NDArray muli(NDArray other) {
      return map(this,
                 other,
                 Operator::multiply,
                 true);
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
      return mapVector(this,
                       other,
                       axis,
                       Operator::multiply,
                       true);
   }

   /**
    * Negates the values in the NDArray
    *
    * @return the new NDArray with negated values
    */
   public NDArray neg() {
      return mapSparse(v -> -v);
   }

   /**
    * Negates the values in the NDArray in-place
    *
    * @return this NDArray
    */
   public NDArray negi() {
      return mapiSparse(v -> -v);
   }

   private NDArray newZeroArray() {
      return getFactory().zeros(shape)
                         .setWeight(weight)
                         .setLabel(label)
                         .setPredicted(predicted);
   }

   /**
    * Norm 1 nd array.
    *
    * @return the nd array
    */
   public NDArray norm1() {
      NDArray[] out = new NDArray[numSlices()];
      forEachSlice((si, n) -> out[si] = DENSE.scalar(valueStream(true).map(Math::abs).sum()));
      return DENSE.fromLayers(numKernels(), numChannels(), out);
   }

   /**
    * Norm 2 nd array.
    *
    * @return the nd array
    */
   public NDArray norm2() {
      return sumOfSquares().mapiSparse(Math::sqrt);
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
    * Number of kernels in the NDArray
    *
    * @return the number of kernels
    */
   public int numKernels() {
      return dimension(Axis.KERNEL);
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
    * Gets the number of slices in the NDArray
    *
    * @return the number of slices
    */
   public int numSlices() {
      return numSlices;
   }

   private NDArray optimum(Axis axis, Optimum optimum) {
      checkArgument(axis.isRowOrColumn(), () -> axisNotSupported(axis));
      int[] newShape = shape();
      newShape[axis.T().ordinal] = 1;
      NDArray out = getFactory().constant((float) optimum.startingValue(), newShape);
      sliceStream().forEach(t -> {
         NDArray outSlice = out.getSlice(t.v1);
         t.v2.iterator().forEachRemaining(e -> {
            int i = axis.select(e.getRow(), e.getColumn());
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

   private NDArray optimumPosition(Axis axis, Optimum optimum) {
      checkArgument(axis.isRowOrColumn(), () -> axisNotSupported(axis));
      final int rows = axis.is(Axis.ROW) ? dimension(axis) : 1;
      final int cols = axis.is(Axis.COLUMN) ? dimension(axis) : 1;
      return sliceUnaryOperation(n -> {
         NDArray out = getFactory().zeros(rows, cols);
         final double[] optimums = new double[Math.max(rows, cols)];
         Arrays.fill(optimums, optimum.startingValue());
         n.forEach(e -> {
            if (optimum.test(e.getValue(), optimums[e.getIndex(axis)])) {
               out.setIndexedValue(0, e.getIndex(axis), e.getIndex(axis.T()));
               optimums[e.getIndex(axis)] = e.getValue();
            }
         });
         return out;
      });
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
                 (v1, v2) -> v2 / v1,
                 false);
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
                       (v1, v2) -> v2 / v1, false);
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
                       (v1, v2) -> v2 / v1,
                       false);
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
                 (v1, v2) -> v2 / v1,
                 false);
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
                       (v1, v2) -> v2 / v1,
                       false);
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
                       (v1, v2) -> v2 / v1,
                       false);
   }

   /**
    * Reshapes the NDArray. (number of slices and length of each slice must remain the same)
    *
    * @param dims The new dimensions
    * @return This NDArray with new shape
    */
   public NDArray reshape(int... dims) {
      checkArgument(dims.length > 0 && dims.length <= 4, () -> invalidNumberOfIndices(dims.length));
      int[] newShape = new int[]{1, 1, 1, 1};
      System.arraycopy(dims, 0, newShape, 0, dims.length);
      int nML = newShape[0] * newShape[1];
      int nSL = newShape[2] * newShape[3];
      checkArgument(nML == matrixLength,
                    () -> "Reshaping must keep same matrix length (" + nML + ") != (" + matrixLength + ").");
      checkArgument(nSL == numSlices,
                    () -> "Reshaping must keep same number of slices (" + nSL + ") != (" + numSlices + ").");
      System.arraycopy(newShape, 0, this.shape, 0, newShape.length);
      this.order = orderOf(this.shape);
      return this;
   }

   /**
    * Row iterator iterator.
    *
    * @param row the row
    * @return the iterator
    */
   public Iterator<Entry> rowIterator(final int row) {
      return new Iterator<Entry>() {
         int slice = 0;
         int column = 0;

         @Override
         public boolean hasNext() {
            return column < numCols() && slice < numSlices();
         }

         @Override
         public Entry next() {
            checkElementIndex(column, numCols());
            Entry e = new Entry(0, toMatrixIndex(row, column));
            column++;
            if (column >= numCols()) {
               slice++;
               column = 0;
            }
            return e;
         }
      };
   }

   private String rowToString(int i, int maxC) {
      StringBuilder builder = new StringBuilder("[");
      builder.append(decimalFormatter.format(get(i, 0)));
      int half = maxC / 2;
      boolean firstHalf = true;
      for (int j = 1; j < numCols(); j++) {
         if (j > half && firstHalf) {
            firstHalf = false;
            int nj = Math.max(numCols() - half, j + 1);
            if (nj > j + 1) {
               builder.append(", ...");
            }
            j = nj;
         }
         builder.append(", ").append(decimalFormatter.format(get(i, j)));
      }
      return builder.append("]").toString();
   }

   /**
    * Subtracts the values in the this NDArray from the other NDArray. Basic broadcasting will occur for scalar, vector,
    * and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be subtracted from
    * @return the new NDArray with the result of other - this
    */
   public NDArray rsub(NDArray other) {
      return map(newZeroArray(),
                 other,
                 (v1, v2) -> v2 - v1,
                 false);
   }

   /**
    * Subtracts each element's value from the given scalar (e.g. scalar - element)
    *
    * @param value the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public NDArray rsub(double value) {
      return mapScalar(newZeroArray(),
                       value,
                       (v1, v2) -> v2 - v1,
                       false);
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
      return mapVector(newZeroArray(),
                       other,
                       axis,
                       (v1, v2) -> v2 - v1,
                       false);
   }

   /**
    * Subtracts each element's value from the given scalar (e.g. scalar - element)  in-place.
    *
    * @param value the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public NDArray rsubi(double value) {
      return mapScalar(this,
                       value,
                       (v1, v2) -> v2 - v1,
                       false);
   }

   /**
    * Subtracts the values in the this NDArray from the other NDArray in-place. Basic broadcasting will occur for
    * scalar, vector, and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be subtracted from
    * @return the new NDArray with the result of other - this
    */
   public NDArray rsubi(NDArray other) {
      return map(this,
                 other,
                 (v1, v2) -> v2 - v1,
                 false);
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
      return mapVector(this,
                       other,
                       axis,
                       (v1, v2) -> v2 - v1,
                       false);
   }

   /**
    * Calculates the dot product of vectors between this NDArray and the given NDArray. The dot product is summed across
    * the slices.
    *
    * @param other the other NDArray to calculate the dot product with
    * @return The sum of the dot products across the slices.
    */
   public double scalarDot(NDArray other) {
      return dot(other).scalarSum();
   }

   /**
    * Finds the max value in the NDArray
    *
    * @return the max value
    */
   public float scalarMax() {
      return optimum(Optimum.MAXIMUM);
   }

   /**
    * Scalar mean double.
    *
    * @return the double
    */
   public double scalarMean() {
      return scalarSum() / length;
   }

   /**
    * Finds the min value in the NDArray
    *
    * @return the min value
    */
   public float scalarMin() {
      return optimum(Optimum.MINIMUM);
   }

   /**
    * Calculates the L1-norm of the NDArray across all slices.
    *
    * @return the L1-norm
    */
   public double scalarNorm1() {
      return norm1().scalarSum();
   }

   /**
    * Calculates the L2-norm (magnitude) of the NDArray across all slices.
    *
    * @return the L1-norm
    */
   public double scalarNorm2() {
      return Math.sqrt(scalarSumOfSquares());
   }

   /**
    * Calculates the sum of all values in the NDArray
    *
    * @return the sum all values
    */
   public double scalarSum() {
      return sum().valueStream(true).sum();
   }

   /**
    * Calculates the sum of squares of all values in the NDArray
    *
    * @return the sum of squares
    */
   public double scalarSumOfSquares() {
      return sumOfSquares().scalarSum();
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
                         v -> predicate.test(v) ? v : 0f,
                         false);
   }

   /**
    * Selects all values matching the given predicate in-place.
    *
    * @param predicate the predicate to test
    * @return this NDArray with values passing the given predicate and zeros elsewhere
    */
   public NDArray selecti(DoublePredicate predicate) {
      return mapOperator(this,
                         v -> predicate.test(v) ? v : 0f,
                         false);
   }

   /**
    * Sets the value at the given indices
    *
    * @param indices the indices
    * @param value   the value
    * @return This NDArray
    */
   public NDArray set(int[] indices, double value) {
      checkArgument(indices.length > 0 && indices.length <= 4, () -> invalidNumberOfIndices(indices.length));
      switch (indices.length) {
         case 1:
            return setIndexedValue(0, indices[0], value);
         case 2:
            return setIndexedValue(0, toMatrixIndex(indices[0], indices[1]), value);
         case 3:
            return setIndexedValue(indices[2], toMatrixIndex(indices[0], indices[1]), value);
         default:
            return setIndexedValue(toSliceIndex(indices[2], indices[3]), toMatrixIndex(indices[0], indices[1]), value);
      }
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
   public NDArray set(int row, int column, int kernel, int channel, double value) {
      return setIndexedValue(toSliceIndex(kernel, channel), toMatrixIndex(row, column), value);
   }

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
      return setIndexedValue(kernel, toMatrixIndex(row, column), value);
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
      return setIndexedValue(0, toMatrixIndex(row, column), value);
   }

   /**
    * Sets the value at the given indices.
    *
    * @param row   the row
    * @param value the value
    * @return This NDArray
    */
   public NDArray set(int row, double value) {
      return setIndexedValue(0, row, value);
   }

   /**
    * Sets indexed value.
    *
    * @param sliceIndex  the slice index
    * @param matrixIndex the matrix index
    * @param value       the value
    * @return the indexed value
    */
   public abstract NDArray setIndexedValue(int sliceIndex, int matrixIndex, double value);

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
      checkArgument(axis.isRowOrColumn(), () -> axisNotSupported(axis));
      int dimLength = dimension(axis.T());
      forEachSlice((i, slice) -> {
         NDArray oSlice = other.getSlice(i);
         for (int j = 0; j < dimLength; j++) {
            int row;
            int col;
            if (axis == Axis.ROW) {
               row = index;
               col = j;
            } else {
               row = j;
               col = index;
            }
            slice.set(row, col, oSlice.get(j));
         }
      });
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
    * Size long.
    *
    * @return the long
    */
   public long size() {
      return length;
   }

   /**
    * Slice nd array.
    *
    * @param from the from
    * @param to   the to
    * @return the nd array
    */
   public abstract NDArray slice(int from, int to);

   /**
    * Slice nd array.
    *
    * @param iFrom the from
    * @param iTo   the to
    * @param jFrom the j from
    * @param jTo   the j to
    * @return the nd array
    */
   public abstract NDArray slice(int iFrom, int iTo, int jFrom, int jTo);

   /**
    * Slice nd array.
    *
    * @param axis    the axis
    * @param indexes the indexes
    * @return the nd array
    */
   public abstract NDArray slice(Axis axis, int... indexes);

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
         sliceStream().forEach(t -> out[t.v1] = function.apply(t.v2, other.getSlice(t.v1)));
      } else {
         sliceStream().forEach(t -> out[t.v1] = function.apply(t.v2, other));
      }
      return getFactory().fromLayers(numKernels(), numChannels(), out);
   }

   /**
    * Slice index stream int stream.
    *
    * @return the int stream
    */
   protected IntStream sliceIndexStream() {
      return IntStream.range(0, numSlices);
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
    * Generates a stream containing tuples of slice index and slice.
    *
    * @return the stream slice index and slice
    */
   public Stream<Tuple2<Integer, NDArray>> sliceStream() {
      return IntStream.range(0, numSlices()).mapToObj(i -> $(i, getSlice(i)));
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
      forEachSlice((si, slice) -> out[si] = function.apply(slice));
      if (out.length == 1) {
         return out[0];
      }
      return getFactory().fromLayers(numKernels(), numChannels(), out);
   }

   /**
    * Sparse column iterator iterator.
    *
    * @param column the column
    * @return the iterator
    */
   public Iterator<Entry> sparseColumnIterator(final int column) {
      return Iterators.filter(columnIterator(column), e -> e.getValue() != 0);
   }

   /**
    * Sparse iterator (Dense implementations will return a dense iterator)
    *
    * @return the non-zero entry iterator
    */
   public Iterator<Entry> sparseIterator() {
      return iterator();
   }

   /**
    * Sparse ordered iterator (Dense implementations will return a dense iterator)
    *
    * @return the sparse entry iterator ordered by slice index, matrix index,
    */
   public Iterator<Entry> sparseOrderedIterator() {
      return sparseIterator();
   }

   /**
    * Sparse row iterator iterator.
    *
    * @param row the row
    * @return the iterator
    */
   public Iterator<Entry> sparseRowIterator(final int row) {
      return Iterators.filter(rowIterator(row), e -> e.getValue() != 0);
   }

   /**
    * Sparse vector iterator iterator.
    *
    * @param index the index
    * @param axis  the axis
    * @return the iterator
    */
   public Iterator<Entry> sparseVectorIterator(int index, Axis axis) {
      checkArgument(axis.isRowOrColumn(), () -> axisNotSupported(axis));
      if (axis == Axis.ROW) {
         return sparseRowIterator(index);
      }
      return sparseColumnIterator(index);
   }

   /**
    * Stream stream.
    *
    * @param sparse the sparse
    * @return the stream
    */
   public Stream<Entry> stream(boolean sparse) {
      return Streams.asStream(() -> iterator(sparse));
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
                 Operator::subtract,
                 false);
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
                       Operator::subtract,
                       false);
   }

   /**
    * Subtracts a scalar value to each element in the NDArray
    *
    * @param value the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public NDArray sub(double value) {
      return mapScalar(newZeroArray(), value, Operator::subtract, false);
   }

   /**
    * Subtracts a scalar value to each element in the NDArray in-place
    *
    * @param value the value to subtract
    * @return the new NDArray with the scalar value subtracted
    */
   public NDArray subi(double value) {
      return mapScalar(this, value, Operator::subtract, false);
   }

   /**
    * Subtracts the values in the other NDArray to this one  in-place. Basic broadcasting will occur for scalar, vector,
    * and matrix NDArrays.
    *
    * @param other the other NDArray whose values will be subtracted
    * @return the new NDArray with the result of this - other
    */
   public NDArray subi(NDArray other) {
      return map(this, other, Operator::subtract,
                 false);
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
                       Operator::subtract, false);
   }

   /**
    * Calculates the sum per slice
    *
    * @return An NDArray of the sums per slice
    */
   public NDArray sum() {
      NDArray[] out = new NDArray[numSlices()];
      forEachSlice((si, n) -> out[si] = getFactory().scalar(n.valueStream(true).sum()));
      return getFactory().fromLayers(numKernels(), numChannels(), out);
   }

   /**
    * Calculates the sum along each axis and per slice
    *
    * @param axis The axis to calculate the sum for
    * @return An NDArray of the sum
    */
   public NDArray sum(Axis axis) {
      checkArgument(axis.isRowOrColumn(), () -> axisNotSupported(axis));
      int[] newShape = shape();
      newShape[axis.T().ordinal] = 1;
      NDArray out = getFactory().zeros(newShape);
      forEachSlice((i, slice) -> {
         for (Iterator<Entry> itr = slice.sparseIterator(); itr.hasNext(); ) {
            Entry e = itr.next();
            out.adjustIndexedValue(i, e.getIndex(axis), e.getValue());
         }
      });
      return out;
   }

   /**
    * Sum of squares nd array.
    *
    * @return the nd array
    */
   public NDArray sumOfSquares() {
      NDArray[] out = new NDArray[numSlices()];
      forEachSlice((si, n) -> out[si] = DENSE.scalar(n.valueStream(true)
                                                      .map(e -> Math.pow(e, 2)).sum()));
      return DENSE.fromLayers(numKernels(), numChannels(), out);
   }

   /**
    * Select nd array.
    *
    * @param predicate  the predicate
    * @param comparison the comparison
    * @return the nd array
    */
   public NDArray test(NDArray predicate, NumericComparison comparison) {
      return map(newZeroArray(),
                 predicate,
                 (v1, v2) -> comparison.compare(v1, v2) ? 1 : 0,
                 false);
   }

   /**
    * Tests the given predicate on the values in the NDArray returning 1 when TRUE and 0 when FALSE
    *
    * @param predicate the predicate to test
    * @return new NDArray with test results
    */
   public NDArray test(DoublePredicate predicate) {
      return mapOperator(newZeroArray(),
                         v -> predicate.test(v) ? 1 : 0f,
                         false);
   }

   /**
    * Selects all values in this NDArray whose corresponding element in the given predicate NDArray is not zero
    * in-place. Basic broadcasting will occur for scalar, vector, and matrix NDArrays.
    *
    * @param predicate  the predicate NDArray test
    * @param comparison the comparison
    * @return this NDArray with values passing the given predicate and zeros elsewhere
    */
   public NDArray testi(NDArray predicate, NumericComparison comparison) {
      return map(this,
                 predicate,
                 (v1, v2) -> comparison.compare(v1, v2) ? 1 : 0,
                 false);
   }

   /**
    * Tests the given predicate on the values in the NDArray returning 1 when TRUE and 0 when FALSE in-place
    *
    * @param predicate the predicate to test
    * @return this with test results
    */
   public NDArray testi(DoublePredicate predicate) {
      return mapOperator(this,
                         v -> predicate.test(v) ? 1 : 0f,
                         false);
   }

   /**
    * To channel int.
    *
    * @param sliceIndex the slice index
    * @return the int
    */
   protected int toChannel(int sliceIndex) {
      return sliceIndex / numKernels();
   }

   /**
    * To column int.
    *
    * @param matrixIndex the matrix index
    * @return the int
    */
   protected int toColumn(int matrixIndex) {
      return matrixIndex / numRows();
   }

   /**
    * To dense dense nd array.
    *
    * @return the dense nd array
    */
   public DenseNDArray toDense() {
      return new DenseNDArray(this);
   }

   /**
    * To double array double [ ].
    *
    * @return the double [ ]
    */
   public double[] toDoubleArray() {
      double[] out = new double[(int) length()];
      forEachSparse(e -> out[(int) e.getIndex()] = e.getValue());
      return out;
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
      float[] out = new float[matrixLength];
      forEachSparse(e -> out[e.matrixIndex] = e.getValue());
      return out;
   }

   /**
    * Generates a JBlas FloatMatrix view of the data
    *
    * @return the float matrix
    */
   public abstract FloatMatrix toFloatMatrix();

   /**
    * To int array int [ ].
    *
    * @param slice the slice
    * @return the int [ ]
    */
   public int[] toIntArray(int slice) {
      int[] out = new int[matrixLength];
      getSlice(slice).forEachSparse(e -> out[e.matrixIndex] = (int) e.getValue());
      return out;
   }

   @Override
   public JsonEntry toJson() {
      JsonEntry ndarray = JsonEntry.object()
                                   .addProperty("shape", shape())
                                   .addProperty("factory", getFactory().name())
                                   .addProperty("weight", weight);
      if (label != null) {
         addLabelProperty(ndarray, "label", label);
      }
      if (predicted != null) {
         addLabelProperty(ndarray, "predicted", predicted);
      }
      JsonEntry array = JsonEntry.array();
      for (int i = 0; i < numSlices(); i++) {
         if (isDense()) {
            array.addValue(getSlice(i).toFloatArray());
         } else {
            JsonEntry map = JsonEntry.object();
            sparseIterator().forEachRemaining(e -> map.addProperty(Integer.toString(e.matrixIndex), e.getValue()));
            array.addValue(map);
         }
      }
      ndarray.addProperty("data", array);
      return ndarray;
   }

   /**
    * To kernel int.
    *
    * @param sliceIndex the slice index
    * @return the int
    */
   protected int toKernel(int sliceIndex) {
      return sliceIndex % numKernels();
   }

   /**
    * To matrix index int.
    *
    * @param row    the row
    * @param column the column
    * @return the int
    */
   protected int toMatrixIndex(int row, int column) {
      return toIndex(row, numRows(), column);
   }

   /**
    * To row int.
    *
    * @param matrixIndex the matrix index
    * @return the int
    */
   protected int toRow(int matrixIndex) {
      return matrixIndex % numRows();
   }

   /**
    * To slice index int.
    *
    * @param kernel  the kernel
    * @param channel the channel
    * @return the int
    */
   protected int toSliceIndex(int kernel, int channel) {
      return toIndex(kernel, numKernels(), channel);
   }

   /**
    * To sparse sparse nd array.
    *
    * @return the sparse nd array
    */
   public SparseNDArray toSparse() {
      return new SparseNDArray(this);
   }

   @Override
   public String toString() {
      return toString(Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE);
   }

   /**
    * To string string.
    *
    * @param maxSlices  the max slices
    * @param maxRows    the max rows
    * @param maxColumns the max columns
    * @return the string
    */
   public String toString(int maxSlices, int maxRows, int maxColumns) {
      StringBuilder builder = new StringBuilder("[");
      builder.append(getSlice(0).matrixToString(maxRows, maxColumns));
      int half = maxSlices / 2;
      boolean firstHalf = true;
      for (int i = 1; i < numSlices(); i++) {
         builder.append(",");
         if (i > half && firstHalf) {
            firstHalf = false;
            int ni = Math.max(numSlices() - half, i + 1);
            if (ni > i + 1) {
               String outDot = StringUtils.repeat(StringUtils.padStart(".", 8, ' '),
                                                  Math.min(numCols(), maxColumns + 2));
               builder.append(System.lineSeparator())
                      .append(System.lineSeparator()).append(outDot)
                      .append(System.lineSeparator()).append(outDot)
                      .append(System.lineSeparator()).append(outDot)
                      .append(System.lineSeparator())
                      .append(System.lineSeparator());
            }
            i = ni;
         }
         builder.append(System.lineSeparator()).append(System.lineSeparator()).append(" ")
                .append(getSlice(0).matrixToString(maxRows, maxColumns));
      }
      return builder.append("]").toString();
   }

   /**
    * To unit vector nd array.
    *
    * @return the nd array
    */
   public NDArray unitize() {
      return div(norm2());
   }

   /**
    * Value stream double stream.
    *
    * @param sparse the sparse
    * @return the double stream
    */
   public DoubleStream valueStream(boolean sparse) {
      return stream(sparse).mapToDouble(Entry::getValue);
   }

   /**
    * Zeroes the entries in this NDArray
    *
    * @return this NDArray with all zeros
    */
   public NDArray zero() {
      return fill(0f);
   }

   private class IndexIterator implements Iterator<Entry> {
      private int matrixIndex = 0;
      private int sliceIndex = 0;

      /**
       * Advance boolean.
       *
       * @return the boolean
       */
      public boolean advance() {
         while ((matrixIndex >= matrixLength && sliceIndex < numSlices)) {
            matrixIndex = 0;
            sliceIndex++;
         }
         return sliceIndex < numSlices;
      }

      @Override
      public boolean hasNext() {
         return advance();
      }

      @Override
      public Entry next() {
         checkState(advance(), "No such element");
         Entry e = new Entry(sliceIndex, matrixIndex);
         matrixIndex++;
         return e;
      }
   }

   /**
    * Defines an entry, or cell, in the NDArray with corresponding indices and value
    */
   public class Entry {

      /**
       * The Matrix index.
       */
      final int matrixIndex;
      /**
       * The Slice index.
       */
      final int sliceIndex;


      /**
       * Instantiates a new Entry.
       *
       * @param sliceIndex  the slice index
       * @param matrixIndex the matrix index
       */
      Entry(int sliceIndex, int matrixIndex) {
         this.sliceIndex = sliceIndex;
         this.matrixIndex = matrixIndex;
      }


      /**
       * Gets channel.
       *
       * @return the channel
       */
      public int getChannel() {
         return toChannel(sliceIndex);
      }

      /**
       * Gets column.
       *
       * @return the column
       */
      public int getColumn() {
         return toColumn(matrixIndex);
      }

      /**
       * Gets index.
       *
       * @return the index
       */
      public long getIndex() {
         return toIndex(matrixIndex, matrixLength, sliceIndex);
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
         return toKernel(sliceIndex);
      }

      /**
       * Gets row.
       *
       * @return the row
       */
      public int getRow() {
         return toRow(matrixIndex);
      }

      /**
       * Gets value.
       *
       * @return the value
       */
      public float getValue() {
         return getIndexedValue(sliceIndex, matrixIndex);
      }

      /**
       * Sets value.
       *
       * @param value the value
       */
      public void setValue(double value) {
         setIndexedValue(sliceIndex, matrixIndex, value);
      }

      /**
       * Matrix index int.
       *
       * @return the int
       */
      public int matrixIndex() {
         return matrixIndex;
      }

      /**
       * Slice index int.
       *
       * @return the int
       */
      public int sliceIndex() {
         return sliceIndex;
      }

      @Override
      public String toString() {
         return "Entry[" +
                   "sliceIndex=" + sliceIndex
                   + ", matrixIndex=" + matrixIndex + "]=" + getValue();
      }

   }


}//END OF NDArray
