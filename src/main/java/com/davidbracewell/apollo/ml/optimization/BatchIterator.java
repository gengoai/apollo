package com.davidbracewell.apollo.ml.optimization;

import com.davidbracewell.apollo.linear.Axis;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import lombok.Getter;
import lombok.val;

import java.io.Serializable;
import java.util.*;

/**
 * @author David B. Bracewell
 */
public class BatchIterator implements Serializable {
   private final List<NDArray> X = new ArrayList<>();
   private final List<NDArray> Y = new ArrayList<>();
   private final int[] indices;
   @Getter
   private final int numLabels;
   @Getter
   private final int numFeatures;

   public BatchIterator(List<NDArray> data, int numLabels, int numFeatures) {
      this.numLabels = numLabels;
      this.numFeatures = numFeatures;
      indices = new int[data.size()];
      for (int i1 = 0; i1 < data.size(); i1++) {
         NDArray vv = data.get(i1);
         indices[i1] = i1;
         val y = NDArrayFactory.defaultFactory().zeros(numLabels, 1);
         y.set((int) vv.getLabelAsDouble(), 0, 1.0);
         Y.add(y);
         X.add(vv);
      }
   }


   public BatchIterator(Dataset<Instance> dataset) {
      List<NDArray> data = dataset.asVectors().collect();
      numLabels = dataset.getLabelEncoder().size();
      numFeatures = dataset.getFeatureEncoder().size();
      indices = new int[data.size()];
      for (int i1 = 0; i1 < data.size(); i1++) {
         NDArray vv = data.get(i1);
         indices[i1] = i1;
         val y = NDArrayFactory.defaultFactory().zeros(numLabels, 1);
         y.set((int) vv.getLabelAsDouble(), 0, 1.0);
         Y.add(y);
         X.add(vv);
      }
   }

   private NDArray create(int rows, List<NDArray> cols, int[] indicies, int start, int end) {
      int numCols = end - start;

      if (numCols == 1) {
         return cols.get(indicies[start]);
      }

      NDArray matrix = NDArrayFactory.defaultFactory().zeros(rows, numCols);
      int mi = 0;
      for (int c = start; c < end; c++) {
         matrix.setVector(mi, cols.get(indicies[c]), Axis.COlUMN);
         mi++;
      }
      return matrix;
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
            val end = Math.min(index + batchSize, indices.length);
            val x_batch = create(numFeatures, X, indices, index, end);
            val y_batch = create(numLabels, Y, indices, index, end);
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
