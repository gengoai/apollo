package com.gengoai.apollo.linear;

import com.gengoai.collection.Streams;
import com.gengoai.conversion.Cast;
import com.gengoai.tuple.Tuple2;
import org.apache.mahout.math.function.IntFloatProcedure;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntFloatHashMap;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

import java.util.BitSet;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static com.gengoai.Validation.*;
import static com.gengoai.tuple.Tuples.$;

/**
 * The type Sparse nd array.
 *
 * @author David B. Bracewell
 */
public class SparseNDArray extends NDArray {
   private final OpenIntFloatHashMap[] data;
   private final BitSet[] bitSet;
//   private final boolean[][] bitSet;

   /**
    * Instantiates a new Sparse nd array.
    *
    * @param dimensions the dimensions
    */
   SparseNDArray(int... dimensions) {
      super(dimensions);
      this.data = new OpenIntFloatHashMap[numSlices()];
      this.bitSet = new BitSet[numSlices()];
//      this.bitSet = new boolean[numSlices()][sliceLength()];
      for (int i = 0; i < numSlices(); i++) {
         this.data[i] = new OpenIntFloatHashMap();
         this.bitSet[i] = new BitSet(sliceLength());
      }
   }


   /**
    * Instantiates a new Sparse nd array.
    *
    * @param copy the copy
    */
   SparseNDArray(NDArray copy) {
      this(copy.shape());
      copy.forEachSlice((si, n) -> n.forEachSparse(e -> setIndexedValue(si, e.matrixIndex, e.getValue())));
   }


   /**
    * Instantiates a new Sparse nd array.
    *
    * @param shape  the shape
    * @param slices the slices
    */
   SparseNDArray(int[] shape, List<SparseNDArray> slices) {
      super(shape);
      this.data = new OpenIntFloatHashMap[numSlices()];
//      this.bitSet = new boolean[numSlices()][sliceLength()];
      this.bitSet = new BitSet[numSlices()];
      for (int i = 0; i < slices.size(); i++) {
         this.data[i] = slices.get(i).data[0];
//         this.bitSet[i] = slices.get(i).bitSet[i];
         this.bitSet[i] = (BitSet) slices.get(i).bitSet[i].clone();
      }
   }

   /**
    * Instantiates a new Sparse nd array.
    *
    * @param rows    the rows
    * @param columns the columns
    * @param matrix  the matrix
    * @param bitSet  the bit set
    */
   SparseNDArray(int rows, int columns, OpenIntFloatHashMap matrix, BitSet bitSet) {
      super(new int[]{rows, columns, 1, 1});
      this.data = new OpenIntFloatHashMap[]{matrix};
      this.bitSet = new BitSet[]{Cast.as(bitSet.clone())};

//      this.bitSet = new boolean[][]{bitSet};

   }

   @Override
   public NDArray adjustIndexedValue(int sliceIndex, int matrixIndex, double value) {
      checkElementIndex(sliceIndex, numSlices(), "Slice");
      checkElementIndex(matrixIndex, sliceLength(), "Matrix");
      if (value == 0) {
         return this;
      }
      data[sliceIndex].adjustOrPutValue(matrixIndex, (float) value, (float) value);
      bitSet[sliceIndex].set(matrixIndex,true);
//      bitSet[sliceIndex][matrixIndex] = true;
      return this;
   }

   @Override
   public NDArray compress() {
      sliceIndexStream().forEach(i -> data[i].trimToSize());
      return this;
   }

   @Override
   public NDArray copyData() {
      return new SparseNDArray(this);
   }

   @Override
   public NDArray fill(double value) {
      if (value == 0) {
         return zero();
      }
      return super.fill(value);
   }

   private void forEachPair(IntFloatProcedure procedure) {
      data[0].forEachPair(procedure);
   }

   @Override
   public NDArrayFactory getFactory() {
      return NDArrayFactory.SPARSE;
   }

   @Override
   public float getIndexedValue(int sliceIndex, int matrixIndex) {
      checkElementIndex(sliceIndex, numSlices(), "Slice");
      checkElementIndex(matrixIndex, sliceLength(), "Matrix");
      if( bitSet[sliceIndex].get(matrixIndex)) {
//      if (bitSet[sliceIndex][matrixIndex]) {
         return data[sliceIndex].get(matrixIndex);
//      }
      }
      return 0f;
   }

   @Override
   public NDArray getSlice(int index) {
      if (order() <= 2) {
         return this;
      }
      return new SparseNDArray(numRows(), numCols(), data[index], bitSet[index]);
      //bitSet[index]);
   }

   @Override
   public boolean isSparse() {
      return true;
   }

   @Override
   public final NDArray mmul(NDArray other) {
      checkArgument(numCols() == other.numRows(),
                    () -> "Number of columns (" + numCols() +
                             ") in this NDArray must equal the number of rows (" +
                             other.numRows() + ") in the other NDArray");
      NDArray toReturn = getFactory().zeros(numRows(), other.numCols(), numKernels(), numChannels());
      for (int sIndex = 0; sIndex < this.data.length; sIndex++) {
         NDArray oth = other.getSlice(sIndex);
         final OpenIntFloatHashMap tSlice = this.data[sIndex];
         final int si = sIndex;
         IntStream.of(tSlice.keys().elements())
                  .parallel()
                  .boxed()
                  .flatMap(mi -> {
                     int row = toRow(mi);
                     int col = toColumn(mi);
                     return Streams.asStream(oth.sparseRowIterator(col))
                                   .map(e -> $(toReturn.toMatrixIndex(row, e.getColumn()),
                                               e.getValue() * tSlice.get(mi)));
                  })
                  .collect(Collectors.groupingBy(Tuple2::getKey,
                                                 Collectors.mapping(Tuple2::getValue,
                                                                    Collectors.reducing(0f, (v1, v2) -> v1 + v2))
                                                )).forEach((i, v) -> toReturn.adjustIndexedValue(si, i, v));
      }
//      forEachSlice((sIndex, slice) -> {
//         NDArray oth = other.getSlice(sIndex);
//         slice.toSparse().forEachPair((index, value) -> {
//            int row = toRow(index);
//            int col = toColumn(index);
//            oth.sparseRowIterator(col)
//               .forEachRemaining(e2 -> toReturn.adjustIndexedValue(sIndex,
//                                                                   toReturn.toMatrixIndex(row, e2.getColumn()),
//                                                                   e2.getValue() * value));
//            return true;
//         });
//      });
      return toReturn;
   }

   @Override
   public NDArray setIndexedValue(int sliceIndex, int matrixIndex, double value) {
      checkElementIndex(sliceIndex, numSlices(), "Slice");
      checkElementIndex(matrixIndex, sliceLength(), "Matrix");
      if (value == 0) {
         data[sliceIndex].removeKey(matrixIndex);
         bitSet[sliceIndex].set(matrixIndex,false);
//         bitSet[sliceIndex][matrixIndex] = false;
      } else {
         bitSet[sliceIndex].set(matrixIndex,true);
//         bitSet[sliceIndex][matrixIndex] = true;
         data[sliceIndex].put(matrixIndex, (float) value);
      }
      return this;
   }

   @Override
   protected void setSlice(int slice, NDArray newSlice) {
      checkArgument(newSlice.order() <= 2, () -> orderNotSupported(newSlice.order()));
      checkElementIndex(slice, numSlices(), "Slice");
      bitSet[slice].clear();
//      Arrays.fill(bitSet[slice], false);
      data[slice].clear();
      newSlice.forEachSparse(e -> setIndexedValue(0, e.matrixIndex, e.getValue()));
   }

   @Override
   public long size() {
      return sliceIndexStream().mapToLong(i -> data[i].size()).sum();
   }

   @Override
   public NDArray slice(int from, int to) {
      int dim = to - from;
      return sliceUnaryOperation(n -> {
         NDArray out = getFactory().zeros(dim);
         for (int i = from; i < to; i++) {
            out.set(i, n.get(i));
         }
         return out;
      });
   }

   @Override
   public NDArray slice(int iFrom, int iTo, int jFrom, int jTo) {
      int rows = iTo - iFrom;
      int cols = jTo - jFrom;
      return sliceUnaryOperation(n -> {
         NDArray out = getFactory().zeros(rows, cols);
         for (int i = iFrom; i < iTo; i++) {
            for (int j = jFrom; j < jTo; j++) {
               out.set(i, j, n.get(i, j));
            }
         }
         return out;
      });
   }

   @Override
   public Iterator<Entry> sparseColumnIterator(int column) {
      return new SparseColumnIndexIterator(column);
   }

   @Override
   public Iterator<Entry> sparseIterator() {
      return new SparseIndexIterator(false);
   }

   @Override
   public Iterator<Entry> sparseOrderedIterator() {
      return new SparseIndexIterator(true);
   }

   @Override
   public Iterator<Entry> sparseRowIterator(int row) {
      return new SparseRowIndexIterator(row);
   }

   @Override
   public DoubleMatrix toDoubleMatrix() {
      return MatrixFunctions.floatToDouble(toFloatMatrix());
   }

   @Override
   public FloatMatrix toFloatMatrix() {
      if (isScalar()) {
         return FloatMatrix.scalar(data[0].get(0));
      }
      checkState(isMatrix());
      return new FloatMatrix(numRows(), numCols(), toFloatArray());
   }

   @Override
   public SparseNDArray toSparse() {
      return this;
   }

   @Override
   public NDArray zero() {
      for (int i = 0; i < numSlices(); i++) {
         data[i].clear();
         bitSet[i].clear();
//         Arrays.fill(bitSet[i], false);
      }
      return this;
   }

   private class SparseIndexIterator implements Iterator<Entry> {
      private final boolean ordered;
      private int matrixIndex = 0;
      private IntArrayList matrixKeys;
      private int sliceIndex = -1;

      private SparseIndexIterator(boolean ordered) {
         this.ordered = ordered;
      }

      /**
       * Advance boolean.
       *
       * @return the boolean
       */
      public boolean advance() {
         while ((matrixKeys == null || matrixIndex >= matrixKeys.size()) && sliceIndex < numSlices()) {
            sliceIndex++;
            if (sliceIndex >= numSlices()) {
               return false;
            }
            setNextKeys();
         }
         return sliceIndex < numSlices();
      }

      @Override
      public boolean hasNext() {
         return advance();
      }

      @Override
      public Entry next() {
         checkState(advance(), "No such element");
         int mi = matrixKeys.get(matrixIndex);
         matrixIndex++;
         return new Entry(sliceIndex, mi);
      }

      private void setNextKeys() {
         matrixKeys = data[sliceIndex].keys();
         matrixIndex = 0;
         if (ordered) {
            matrixKeys.sort();
         }
      }
   }

   private class SparseRowIndexIterator implements Iterator<Entry> {
      private int col = 0;
      private int sliceIndex = 0;
      private int lastMatrixIndex;
      private int startingMatrixIndex;
      private byte STATE = 0; // 0 - dirty, 1 - has next, 2 - end of iterator

      private SparseRowIndexIterator(int row) {
         this.lastMatrixIndex = toMatrixIndex(row, 0);
         this.startingMatrixIndex = this.lastMatrixIndex;
      }

      /**
       * Advance boolean.
       *
       * @return the boolean
       */
      public boolean advance() {
         switch (STATE) {
            case 1:
               return true;
            case 2:
               return false;
            default:
               return computeNext();
         }
      }

      private boolean advanceColumn() {
         while (col < numCols() && !bitSet[sliceIndex].get(lastMatrixIndex)) {
//                   && !bitSet[sliceIndex][lastMatrixIndex]) {
            col++;
            lastMatrixIndex += numRows();
         }
         STATE = (col < numCols()) ? (byte) 1 : 0;
         return STATE == 1;
      }

      private boolean computeNext() {
         while (STATE != 2 && !advanceColumn()) {
            sliceIndex++;
            col = 0;
            lastMatrixIndex = startingMatrixIndex;
            STATE = (sliceIndex < numSlices()) ? (byte) 0 : 2;
         }
         return STATE == 1;
      }

      @Override
      public boolean hasNext() {
         return advance();
      }

      @Override
      public Entry next() {
         checkState(advance(), "No such element");
         Entry e = new Entry(sliceIndex, lastMatrixIndex);
         col++;
         lastMatrixIndex += numRows();
         STATE = 0;
         return e;
      }
   }

   private class SparseColumnIndexIterator implements Iterator<Entry> {
      private final int col;
      private int row = 0;
      private int sliceIndex = 0;
      private int mi;
      private byte STATE = 0; // 0 - dirty, 1 - has next, 2 - end of iterator

      private SparseColumnIndexIterator(int col) {
         this.col = col;
         this.mi = numRows() * col;
      }

      /**
       * Advance boolean.
       *
       * @return the boolean
       */
      public boolean advance() {
         switch (STATE) {
            case 1:
               return true;
            case 2:
               return false;
            default:
               return computeNext();
         }
      }

      private boolean advanceRow() {
         while (row < numRows() && !bitSet[sliceIndex].get(mi)) {// && !bitSet[sliceIndex][mi]) {
            row++;
            mi++;
         }
         STATE = (row < numRows()) ? (byte) 1 : 0;
         return STATE == 1;
      }

      private boolean computeNext() {
         while (STATE != 2 && !advanceRow()) {
            sliceIndex++;
            row = 0;
            mi = numRows() * col;
            STATE = (sliceIndex < numSlices()) ? (byte) 0 : 2;
         }
         return STATE == 1;
      }

      @Override
      public boolean hasNext() {
         return advance();
      }

      @Override
      public Entry next() {
         checkState(advance(), "No such element");
         Entry e = new Entry(sliceIndex, mi);
         row++;
         mi++;
         STATE = 0;
         return e;
      }
   }
}//END OF SparseNDArray
