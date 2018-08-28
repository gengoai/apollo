package com.gengoai.apollo.linear;

import com.gengoai.Copyable;
import com.gengoai.collection.Iterators;
import com.gengoai.collection.Streams;
import com.gengoai.conversion.Cast;
import com.gengoai.json.JsonEntry;
import com.gengoai.json.JsonSerializable;
import com.gengoai.math.Math2;
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
import static com.gengoai.collection.Iterators.zipWithIndex;
import static com.gengoai.tuple.Tuples.$;

/**
 * The type Nd array.
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
      if (this.shape[Axis.CHANNEL.ordinal] > 1) {
         this.order = 4;
      } else if (this.shape[Axis.KERNEL.ordinal] > 1) {
         this.order = 3;
      } else if (this.shape[Axis.ROW.ordinal] > 1 && this.shape[Axis.COLUMN.ordinal] > 1) {
         this.order = 2;
      } else if (this.shape[Axis.ROW.ordinal] > 1 || this.shape[Axis.COLUMN.ordinal] > 1) {
         this.order = 1;
      } else {
         this.order = 0;
      }
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

   static String invalidNumberOfIndices(int length) {
      return "Invalid number of indices (" + length + ")";
   }

   static String orderNotSupported(int order) {
      return "Order (" + order + ") is not supported.";
   }

   static String sliceOutBounds(int slice, int numSlices) {
      return "Slice index (" + slice + ") out of bounds [0, " + numSlices + ")";
   }

   static String lengthMismatch(int l1, int l2) {
      return "Length mismatch (" + l1 + ") != (" + l2 + ")";
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
    * Calculates a dimension-2 major index (e.g. column major) given two axis indices and their dimensions
    *
    * @param ax1    the index along axis 1
    * @param dimAx1 the dimension of axis 1
    * @param ax2    the index along axis 2
    * @param dimAx2 the dimension of axis 2
    * @return the encoded index
    */
   public static int toIndex(int ax1, int dimAx1, int ax2, int dimAx2) {
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
    * Adjust or put indexed value nd array.
    *
    * @param sliceIndex  the slice index
    * @param matrixIndex the matrix index
    * @param value       the value
    * @return the nd array
    */
   protected abstract NDArray adjustOrPutIndexedValue(int sliceIndex, int matrixIndex, double value);

   /**
    * Calculates the index of the maximum along each row or column based on the given axis.
    *
    * @param axis the axis (row/column) to calculate the max for
    * @return array of int array of row/column indexes relating to max values per slice
    */
   public int[][] argMax(Axis axis) {
      return argOptimum(axis,
                        Optimum.MAXIMUM);
   }

   /**
    * Calculates the index of the minimum along each row or column based on the given axis.
    *
    * @param axis the axis (row/column) to calculate the minimum for
    * @return array of int array of row/column indexes relating to minimum values per slice
    */
   public int[][] argMin(Axis axis) {
      return argOptimum(axis,
                        Optimum.MINIMUM);
   }

   private int[][] argOptimum(Axis axis, Optimum optimum) {
      checkArgument(axis.isRowOrColumn(), () -> axisNotSupported(axis));
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
    * Copy data nd array.
    *
    * @return the nd array
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
      return adjustOrPutIndexedValue(toSliceIndex(kernel, channel), toMatrixIndex(row, column), -value);
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
      return adjustOrPutIndexedValue(kernel, toMatrixIndex(row, column), -value);
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
      return adjustOrPutIndexedValue(0, toMatrixIndex(row, column), -value);
   }

   /**
    * Decrements the value of the NDArray at the given indices.
    *
    * @param row   the row
    * @param value the value to decrement by
    * @return This NDArray
    */
   public NDArray decrement(int row, double value) {
      return adjustOrPutIndexedValue(0, row, -value);
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
         slice.forEachSparse(e -> out.adjustOrPutIndexedValue(si,
                                                              e.matrixIndex,
                                                              os.get(e.matrixIndex) * e.getValue()));
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
      checkArgument(matrixLength == other.matrixLength, () -> lengthMismatch(matrixLength, other.matrixLength));
      return (float) sliceStream().mapToDouble(t -> {
         NDArray os = other.getSlice(t.v1);
         double dot = 0d;
         NDArray small = t.v2.size() > os.size() ? os : t.v2;
         NDArray big = t.v2.size() > os.size() ? t.v2 : os;
         for (Iterator<Entry> itr = small.sparseIterator(); itr.hasNext(); ) {
            Entry e = itr.next();
            dot += e.getValue() * big.getIndexedValue(0, e.matrixIndex);
         }
         return dot;
      }).sum();
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
      iterator().forEachRemaining(e -> e.setValue(value));
      return this;
   }

   /**
    * Fills the NDArray with values generated from the given double supplier
    *
    * @param supplier the supplier to use to set all cells in the NDArray
    * @return This NDArray
    */
   public NDArray fill(DoubleSupplier supplier) {
      iterator().forEachRemaining(e -> e.setValue(supplier.getAsDouble()));
      return this;
   }

   /**
    * Processes each slice of the NDArray using the given NDArray consumer
    *
    * @param sliceConsumer the slice consumer
    */
   public void forEachSlice(BiConsumer<Integer, NDArray> sliceConsumer) {
      for (int i = 0; i < slices(); i++) {
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
    * Gets indexed value.
    *
    * @param sliceIndex  the slice index
    * @param matrixIndex the matrix index
    * @return the indexed value
    */
   public abstract float getIndexedValue(int sliceIndex, int matrixIndex);

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
      return adjustOrPutIndexedValue(toSliceIndex(kernel, channel), toMatrixIndex(row, column), value);
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
      return adjustOrPutIndexedValue(kernel, toMatrixIndex(row, column), value);
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
      return adjustOrPutIndexedValue(0, toMatrixIndex(row, column), value);
   }

   /**
    * Increments the value of the NDArray at the given indices.
    *
    * @param row   the row
    * @param value the value to increment by
    * @return This NDArray
    */
   public NDArray increment(int row, double value) {
      return adjustOrPutIndexedValue(0, row, value);
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
      return new IndexIterator();
   }

   private Iterator<Entry> iterator(boolean sparse) {
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
    * Calculates the L1-norm of the NDArray across all slices. See {@link #sliceNorm1()} to calculate per slice.
    *
    * @return the L1-norm
    */
   public float norm1() {
      return (float) sliceStream().mapToDouble(t -> Streams.asStream(t.v2)
                                                           .mapToDouble(e -> Math.abs(e.getValue()))
                                                           .sum()).sum();
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
            return column < numCols() && slice < slices();
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
    * Selects all values in this NDArray whose corresponding element in the given predicate NDArray is not zero. Basic
    * broadcasting will occur for scalar, vector, and matrix NDArrays.
    *
    * @param predicate the predicate NDArray test
    * @return new NDArray with values passing the given predicate and zeros elsewhere
    */
   public NDArray select(NDArray predicate) {
      return map(newZeroArray(),
                 predicate,
                 (v1, v2) -> v2 != 0 ? v1 : 0f,
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
    * Selects all values in this NDArray whose corresponding element in the given predicate NDArray is not zero
    * in-place. Basic broadcasting will occur for scalar, vector, and matrix NDArrays.
    *
    * @param predicate the predicate NDArray test
    * @return this NDArray with values passing the given predicate and zeros elsewhere
    */
   public NDArray selecti(NDArray predicate) {
      return map(this,
                 predicate,
                 (v1, v2) -> v2 != 0 ? v1 : 0f,
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
    * Shape equals boolean.
    *
    * @param other the other
    * @return the boolean
    */
   public boolean shapeEquals(NDArray other) {
      return Arrays.equals(shape, other.shape);
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
      return IntStream.range(0, slices()).mapToObj(i -> $(i, getSlice(i)));
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
//      for (int i = 0; i < slices(); i++) {
//         sliceConsumer.accept(i, getMatrix(i));
//      }
      NDArray[] out = new NDArray[numSlices];
      forEachSlice((si, slice) -> out[si] = function.apply(slice));
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

   public Iterator<Entry> sparseVectorIterator(int index, Axis axis) {
      checkArgument(axis.isRowOrColumn(), () -> axisNotSupported(axis));
      if (axis == Axis.ROW) {
         return sparseRowIterator(index);
      }
      return sparseColumnIterator(index);
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
      checkArgument(axis.isRowOrColumn(), () -> axisNotSupported(axis));
      int[] newShape = shape();
      newShape[axis.T().ordinal] = 1;
      NDArray out = getFactory().zeros(newShape);
      forEachSlice((i, slice) -> {
         for (Iterator<Entry> itr = slice.sparseIterator(); itr.hasNext(); ) {
            Entry e = itr.next();
            out.adjustOrPutIndexedValue(i, e.getIndex(axis), e.getValue());
         }
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
      return mapOperator(newZeroArray(),
                         v -> predicate.test(v) ? 1 : 0f,
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
   public int toChannel(int sliceIndex) {
      return sliceIndex / numKernels();
   }

   /**
    * To column int.
    *
    * @param matrixIndex the matrix index
    * @return the int
    */
   public int toColumn(int matrixIndex) {
      return matrixIndex / numRows();
   }

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
      sparseIterator().forEachRemaining(e -> out[(int) e.getIndex()] = e.getValue());
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
      for (int i = 0; i < slices(); i++) {
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
   public int toKernel(int sliceIndex) {
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
      return toIndex(row, numRows(), column, numCols());
   }

   /**
    * To row int.
    *
    * @param matrixIndex the matrix index
    * @return the int
    */
   public int toRow(int matrixIndex) {
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
      return toIndex(kernel, shape[Axis.KERNEL.ordinal],
                     channel, shape[Axis.CHANNEL.ordinal]);
   }

   public SparseNDArray toSparse() {
      return new SparseNDArray(this);
   }

   private String toString(FloatMatrix matrix) {
      return matrix.toString("%f", "[", "]", ", ", "],\n  [");
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
      for (int i = 1; i < slices(); i++) {
         builder.append(",");
         if (i > half && firstHalf) {
            firstHalf = false;
            int ni = Math.max(slices() - half, i + 1);
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
   public NDArray toUnitVector() {
      checkArgument(isVector(), "NDArray must be a vector");
      float mag = norm2();
      return div(mag);
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
         return toIndex(matrixIndex, matrixLength, sliceIndex, numSlices);
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
