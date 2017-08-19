package com.davidbracewell.apollo.ml.classification.nn.slt;

import com.davidbracewell.apollo.linalg.DenseFloatMatrix;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.tuple.Tuple2;
import lombok.Getter;
import lombok.val;

import java.io.Serializable;
import java.util.*;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
public class MatrixTrainSet implements Serializable {
   private final List<DenseFloatMatrix> X = new ArrayList<>();
   private final List<DenseFloatMatrix> Y = new ArrayList<>();
   private final int[] indices;
   @Getter
   private final int numLabels;
   @Getter
   private final int numFeatures;

   public MatrixTrainSet(Dataset<Instance> dataset) {
      List<Vector> data = dataset.asVectors().collect();
      numLabels = dataset.getLabelEncoder().size();
      numFeatures = dataset.getFeatureEncoder().size();
      indices = new int[data.size()];
      for (int i1 = 0; i1 < data.size(); i1++) {
         Vector vv = data.get(i1);
         indices[i1] = i1;
         val y = DenseFloatMatrix.zeros(numLabels, 1);
         y.set((int) vv.getLabelAsDouble(), 0, 1.0);
         Y.add(y);
         X.add(new DenseFloatMatrix(1, vv.dimension(), vv.toFloatArray()));
      }
   }

   private DenseFloatMatrix create(int rows, List<DenseFloatMatrix> cols, int[] indicies, int start, int end) {
      int numCols = end - start;
      DenseFloatMatrix matrix = DenseFloatMatrix.zeros(rows, numCols);
      int mi = 0;
      for (int c = start; c < end; c++) {
         matrix.setColumn(mi, cols.get(indicies[c]));
         mi++;
      }
      return matrix;
   }

   public Iterator<Tuple2<Matrix, Matrix>> iterator(final int batchSize) {
      return new Iterator<Tuple2<Matrix, Matrix>>() {
         private int index = 0;

         @Override
         public boolean hasNext() {
            return index < indices.length;
         }

         @Override
         public Tuple2<Matrix, Matrix> next() {
            if (!hasNext()) {
               throw new NoSuchElementException();
            }
            val end = Math.min(index + batchSize, indices.length);
            val x_batch = create(numFeatures, X, indices, index, end);
            val y_batch = create(numLabels, Y, indices, index, end);
            index = end;
            return $(x_batch, y_batch);
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


}// END OF MatrixTrainSet
