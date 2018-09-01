package com.gengoai.apollo.linear;

import com.gengoai.conversion.Cast;
import org.apache.mahout.math.function.IntFloatProcedure;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntFloatHashMap;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.function.Function;

import static com.gengoai.Validation.*;

/**
 * @author David B. Bracewell
 */
public class SparseNDArray extends NDArray {
   private final OpenIntFloatHashMap[] data;
   private final boolean[][] bitSet;

   SparseNDArray(int... dimensions) {
      super(dimensions);
      this.data = new OpenIntFloatHashMap[numSlices()];
      this.bitSet = new boolean[numSlices()][sliceLength()];
      for (int i = 0; i < numSlices(); i++) {
         this.data[i] = new OpenIntFloatHashMap();
      }
   }


   SparseNDArray(NDArray copy) {
      this(copy.shape());
      copy.forEachSlice((si, n) -> n.forEachSparse(e -> setIndexedValue(si, e.matrixIndex, e.getValue())));
   }


   SparseNDArray(int[] shape, List<SparseNDArray> slices) {
      super(shape);
      this.data = new OpenIntFloatHashMap[numSlices()];
      this.bitSet = new boolean[numSlices()][sliceLength()];
      for (int i = 0; i < slices.size(); i++) {
         this.data[i] = slices.get(i).data[0];
         this.bitSet[i] = slices.get(i).bitSet[i];
      }
   }

   SparseNDArray(int rows, int columns, OpenIntFloatHashMap matrix, boolean[] bitSet) {
      super(new int[]{rows, columns, 1, 1});
      this.data = new OpenIntFloatHashMap[]{matrix};
      this.bitSet = new boolean[][]{bitSet};

   }

   @Override
   public NDArray adjustIndexedValue(int sliceIndex, int matrixIndex, double value) {
      checkElementIndex(sliceIndex, numSlices(), "Slice");
      checkElementIndex(matrixIndex, sliceLength(), "Matrix");
      if (value == 0) {
         return this;
      }
      data[sliceIndex].adjustOrPutValue(matrixIndex, (float) value, (float) value);
      bitSet[sliceIndex][matrixIndex] = true;
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

   protected void forEachPair(IntFloatProcedure procedure) {
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
      if (bitSet[sliceIndex][matrixIndex]) {
         return data[sliceIndex].get(matrixIndex);
      }
      return 0f;
   }

   @Override
   public NDArray getSlice(int index) {
      if (order() <= 2) {
         return this;
      }
      return new SparseNDArray(numRows(), numCols(), data[index], bitSet[index]);
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
      forEachSlice((sIndex, slice) -> {
         NDArray oth = other.getSlice(sIndex);
         slice.toSparse().forEachPair((index, value) -> {
            int row = toRow(index);
            int col = toColumn(index);
            oth.sparseRowIterator(col)
               .forEachRemaining(e2 -> toReturn.adjustIndexedValue(sIndex,
                                                                   toReturn.toMatrixIndex(row, e2.getColumn()),
                                                                   e2.getValue() * value));
            return true;
         });
      });
      return toReturn;
   }

   @Override
   public NDArray setIndexedValue(int sliceIndex, int matrixIndex, double value) {
      checkElementIndex(sliceIndex, numSlices(), "Slice");
      checkElementIndex(matrixIndex, sliceLength(), "Matrix");
      if (value == 0) {
         data[sliceIndex].removeKey(matrixIndex);
         bitSet[sliceIndex][matrixIndex] = false;
      } else {
         bitSet[sliceIndex][matrixIndex] = true;
         data[sliceIndex].put(matrixIndex, (float) value);
      }
      return this;
   }

   @Override
   protected void setSlice(int slice, NDArray newSlice) {
      checkArgument(newSlice.order() <= 2, () -> orderNotSupported(newSlice.order()));
      checkElementIndex(slice, numSlices(), "Slice");
      Arrays.fill(bitSet[slice], false);
      data[slice].clear();
      newSlice.forEachSparse(e -> setIndexedValue(0, e.matrixIndex, e.getValue()));
   }

   @Override
   public long size() {
      return sliceIndexStream().mapToLong(i -> data[i].size()).sum();
   }

   @Override
   public NDArray slice(int from, int to) {
      return null;
   }

   @Override
   public NDArray slice(int iFrom, int iTo, int jFrom, int jTo) {
      return null;
   }

   @Override
   public NDArray slice(Axis axis, int... indexes) {
      return null;
   }

   @Override
   public NDArray sliceUnaryOperation(Function<NDArray, NDArray> function) {
      SparseNDArray[] out = new SparseNDArray[numSlices()];
      sliceStream().forEach(t -> {
         NDArray i = function.apply(t.v2);
         if (i instanceof SparseNDArray) {
            out[t.v1] = Cast.as(i);
         } else {
            out[t.v1] = new SparseNDArray(out[t.v1]);
         }
      });
      return new SparseNDArray(new int[]{out[0].numRows(), out[0].numCols(), numKernels(), numChannels()},
                               Arrays.asList(out));
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
         Arrays.fill(bitSet[i], false);
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


      private SparseRowIndexIterator(int row) {
         this.lastMatrixIndex = toMatrixIndex(row, 0);
         this.startingMatrixIndex = this.lastMatrixIndex;
      }

      public boolean advance() {
         while (sliceIndex < numSlices() && !advanceColumn()) {
            sliceIndex++;
            col = 0;
            lastMatrixIndex = startingMatrixIndex;
         }
         return sliceIndex < numSlices();
      }

      private boolean advanceColumn() {
         while (col < numCols()
                   && !bitSet[sliceIndex][lastMatrixIndex]) {
            col++;
            lastMatrixIndex += numRows();
         }
         return col < numCols();
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

      private boolean computeNext() {
         while (STATE != 2 && !advanceRow()) {
            sliceIndex++;
            row = 0;
            mi = numRows() * col;
            STATE = (sliceIndex < numSlices()) ? (byte) 0 : 2;
         }
         return STATE == 1;
      }

      private boolean advanceRow() {
         while (row < numRows() && !bitSet[sliceIndex][mi]) {
            row++;
            mi++;
         }
         STATE = (row < numRows()) ? (byte) 1 : 0;
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
