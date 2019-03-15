package com.gengoai.apollo.optimization;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;

import java.io.Serializable;
import java.util.*;

/**
 * @author David B. Bracewell
 */
public class BatchIterator implements Serializable {
   private final List<NDArray> X = new ArrayList<>();
   private final List<NDArray> Y = new ArrayList<>();
   private final int[] indices;
   private final int numLabels;
   private final int numFeatures;

   public int getNumLabels() {
      return numLabels;
   }

   public int getNumFeatures() {
      return numFeatures;
   }


   public BatchIterator(List<NDArray> data, int numLabels, int numFeatures) {
      this.numLabels = numLabels;
      this.numFeatures = numFeatures;
      indices = new int[data.size()];
      for (int i1 = 0; i1 < data.size(); i1++) {
         NDArray vv = data.get(i1);
         indices[i1] = i1;
         NDArray y = vv.getLabelAsNDArray(numLabels);
         Y.add(y);
         X.add(vv);
      }
   }

   private NDArray create(int rows, List<NDArray> cols, int[] indicies, int start, int end) {
      int numCols = end - start;

      if (numCols == 1) {
         return cols.get(indicies[start]);
      }

      NDArray[] concat = new NDArray[numCols];
      int mi = 0;
      for (int c = start; c < end; c++) {
         concat[mi] = cols.get(indicies[c]);
         mi++;
      }

      return NDArrayFactory.ND.hstack(concat);
   }

   public Iterator<NDArray> iterator(final int batchSize) {
      return new Iterator<NDArray>() {
         private int index = 0;

         @Override
         public boolean hasNext() {
            return index < indices.length;
         }

         @Override
         public NDArray next() {
            if (!hasNext()) {
               throw new NoSuchElementException();
            }
            int end = Math.min(index + batchSize, indices.length);
            NDArray x_batch = create(numFeatures, X, indices, index, end);
            NDArray y_batch = create(numLabels, Y, indices, index, end);
            x_batch.setLabel(y_batch);
            index = end;
            return x_batch;
         }
      };
   }

   public void shuffle() {
      Random rnd = new Random();
      for (int i = 0; i < indices.length; i++) {
         int idx = rnd.nextInt(indices.length);
         int temp = indices[i];
         indices[i] = indices[idx];
         indices[idx] = temp;
      }
   }

   public int size() {
      return X.size();
   }

   public Iterator<List<NDArray>> threadIterator(final int batchSize, final int numberOfThreads) {
      final int numberOfBatches = (size() / batchSize);
      final int batchesPerThread = numberOfBatches / numberOfThreads;
      return new Iterator<List<NDArray>>() {
         final Iterator<NDArray> itr = iterator(batchSize);
         int thread = 1;
         List<NDArray> b = null;

         private boolean forward() {
            if (b == null && itr.hasNext()) {
               b = new ArrayList<>();
               int i = 0;
               while (itr.hasNext() && (i < batchesPerThread || thread == numberOfThreads)) {
                  i++;
                  b.add(itr.next());
               }
               thread++;
            }
            return b != null && !b.isEmpty();
         }

         @Override
         public boolean hasNext() {
            return forward();
         }

         @Override
         public List<NDArray> next() {
            if (!forward()) {
               throw new NoSuchElementException();
            }
            List<NDArray> toReturn = b;
            b = null;
            return toReturn;
         }
      };
   }


}// END OF BatchIterator
