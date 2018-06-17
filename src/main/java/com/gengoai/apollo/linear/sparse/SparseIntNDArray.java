package com.gengoai.apollo.linear.sparse;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import org.apache.mahout.math.function.IntDoubleProcedure;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntIntHashMap;

/**
 * @author David B. Bracewell
 */
public class SparseIntNDArray extends SparseNDArray {
   private static final long serialVersionUID = 1L;
   private final OpenIntIntHashMap storage = new OpenIntIntHashMap();

   public SparseIntNDArray(int nRows, int nCols) {
      super(nRows, nCols);
   }

   @Override
   protected double adjustOrPutValue(int index, double amount) {
      return storage.adjustOrPutValue(index, (int) amount, (int) amount);
   }

   @Override
   public NDArray compress() {
      storage.trimToSize();
      return this;
   }

   @Override
   protected void forEachPair(IntDoubleProcedure procedure) {
      storage.forEachPair((index, value) -> {
         procedure.apply(index, value);
         return true;
      });
   }

   @Override
   public double get(int index) {
      return storage.get(index);
   }


   @Override
   public NDArrayFactory getFactory() {
      return NDArrayFactory.SPARSE_INT;
   }

   @Override
   protected IntArrayList nonZeroIndexes() {
      return storage.keys();
   }

   @Override
   protected void removeIndex(int index) {
      storage.removeKey(index);
   }

   @Override
   protected void setValue(int index, double value) {
      storage.put(index, (int) value);
   }

   @Override
   public int size() {
      return storage.size();
   }


}//END OF SparseFloatNDArray
