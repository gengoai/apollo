package com.gengoai.apollo.linear.v2;

import org.apache.mahout.math.map.OpenIntFloatHashMap;
import org.jblas.DoubleMatrix;
import org.jblas.FloatMatrix;
import org.jblas.MatrixFunctions;

import java.util.List;
import java.util.Objects;

import static com.gengoai.Validation.checkArgument;
import static com.gengoai.Validation.checkState;

/**
 * @author David B. Bracewell
 */
public class SparseNDArray extends NDArray {
   private final OpenIntFloatHashMap[] data;

   SparseNDArray(int... dimensions) {
      super(dimensions);
      this.data = new OpenIntFloatHashMap[slices()];
      for (int i = 0; i < slices(); i++) {
         this.data[i] = new OpenIntFloatHashMap();
      }
   }

   SparseNDArray(NDArray copy) {
      this(copy.shape());
      copy.sliceStream().forEach(t -> {
         int index = t.v1;
         NDArray slice = t.v2.slice(index);
         slice.forEachSparse(e -> this.data[index].put(e.matrixIndex(), e.getValue()));
      });
   }


   SparseNDArray(int[] shape, List<SparseNDArray> slices) {
      this(shape);
      for (int i = 0; i < slices.size(); i++) {
         this.data[i] = slices.get(i).data[0];
      }
   }

   SparseNDArray(int rows, int columns, OpenIntFloatHashMap matrix) {
      this(rows, columns);
      this.data[0] = matrix;
   }


   @Override
   public NDArray copy() {
      return new SparseNDArray(this).setWeight(getWeight())
                                    .setLabel(getLabel())
                                    .setPredicted(getPredicted());
   }

   @Override
   public boolean isSparse() {
      return true;
   }

   public float get(int index) {
      if (order() <= 2) {
         return data[0].get(index);
      }
      return get(fromIndex(index, shape));
   }

   public float get(int row, int column) {
      return data[0].get(toMatrixIndex(row, column));
   }

   @Override
   public float get(int... indices) {
      switch (indices.length) {
         case 1:
            if (order() <= 2) {
               return data[0].get(indices[0]);
            }
            return get(fromIndex(indices[0], shape));
         case 2:
            return data[0].get(toMatrixIndex(indices[0], indices[1]));
         case 3:
            return data[indices[2]].get(toMatrixIndex(indices[0], indices[1]));
         case 4:
            return data[toSliceIndex(indices[2], indices[3])].get(toMatrixIndex(indices[0], indices[1]));
      }
      throw new IllegalArgumentException("Too many indices");
   }

   @Override
   public NDArrayFactory getFactory() {
      return NDArrayFactory.SPARSE;
   }

   @Override
   public NDArray set(int row, int column, int kernel, int channel, float value) {
      int si = toSliceIndex(kernel, channel);
      int mi = toMatrixIndex(row, column);
      if (value == 0) {
         data[si].removeKey(mi);
      } else {
         data[si].put(mi, value);
      }
      return this;
   }

   @Override
   protected void setSlice(int slice, NDArray newSlice) {
      checkArgument(newSlice.order() <= 2, "Order (" + newSlice.order() + ") is not supported");
      newSlice.forEach(e -> set(e.getIndicies(), e.getValue()));
   }

   @Override
   public NDArray slice(int index) {
      return new SparseNDArray(numRows(), numCols(), data[index]);
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
      return new FloatMatrix(toFloatArray());
   }


   @Override
   public boolean equals(Object obj) {
      if (this == obj) {return true;}
      if (obj == null || getClass() != obj.getClass()) {return false;}
      if (!super.equals(obj)) {return false;}
      final SparseNDArray other = (SparseNDArray) obj;
      return Objects.deepEquals(this.data, other.data);
   }
}//END OF SparseNDArray
