package com.davidbracewell.apollo.linear.sparse;

import com.davidbracewell.apollo.linear.NDArrayFactory;
import org.apache.mahout.math.function.IntDoubleProcedure;
import org.apache.mahout.math.list.IntArrayList;
import org.apache.mahout.math.map.OpenIntDoubleHashMap;

/**
 * @author David B. Bracewell
 */
public class SparseDoubleNDArray extends SparseNDArray {
   private static final long serialVersionUID = 1L;
   private final OpenIntDoubleHashMap storage = new OpenIntDoubleHashMap();

   public SparseDoubleNDArray(int nRows, int nCols) {
      super(nRows, nCols);
   }

   @Override
   protected double adjustOrPutValue(int index, double amount) {
      return storage.adjustOrPutValue(index, amount, amount);
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
      return NDArrayFactory.SPARSE_DOUBLE;
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
      storage.put(index, value);
   }

   @Override
   public int size() {
      return storage.size();
   }


}//END OF SparseDoubleNDArray
