package com.davidbracewell.apollo.ml.classification.nn;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.guava.common.base.Stopwatch;
import lombok.val;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * @author David B. Bracewell
 */
public class MLPLearner extends ClassifierLearner {
   static float correct(DoubleMatrix predicted, DoubleMatrix gold) {
      int[] pMax = predicted.columnArgmaxs();
      int[] gMax = gold.columnArgmaxs();
      float correct = 0;
      for (int i = 0; i < pMax.length; i++) {
         if (pMax[i] == gMax[i]) {
            correct++;
         }
      }
      return correct;
   }

   public static DoubleMatrix rand(int numRows, int numCols) {
      float max = (float) (Math.sqrt(6.0) / Math.sqrt(numCols + numRows));
      float min = -max;
      DoubleMatrix f = DoubleMatrix.zeros(numRows, numCols);
      for (int i = 0; i < f.length; i++) {
         f.data[i] = (float) (min + (max - min) * Math.random());
      }
      return f;
   }

   private static int[] shuffle(int[] array) {
      Random rnd = new Random();
      for (int i = 0; i < array.length; i++) {
         int idx = rnd.nextInt(array.length);
         int temp = array[i];
         array[i] = array[idx];
         array[idx] = temp;
      }
      return array;
   }

   @Override
   protected void resetLearnerParameters() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      MLP mlp = new MLP(this);
      List<Vector> data = dataset.asVectors().collect();
      int nL = dataset.getLabelEncoder().size();
      int nF = dataset.getFeatureEncoder().size();
      int X_size = data.size();
      DoubleMatrix allX = DoubleMatrix.zeros(nF, X_size);
      DoubleMatrix allY = DoubleMatrix.zeros(nL, X_size);
      int[] shuffle = new int[X_size];
      for (int i1 = 0; i1 < data.size(); i1++) {
         Vector vv = data.get(i1);
         shuffle[i1] = i1;
         allY.put((int) vv.getLabelAsDouble(), i1, 1.0f);
         allX.putColumn(i1, new DoubleMatrix(1, vv.dimension(), vv.toArray()));
      }
      try {
         dataset.close();
      } catch (Exception e) {
         e.printStackTrace();
      }
      data = null;
      mlp.w1 = rand(300, nF);
      mlp.b1 = DoubleMatrix.zeros(300, 1);
      mlp.w2 = rand(nL, 300);
      mlp.b2 = DoubleMatrix.zeros(nL, 1);

      final int batch_size = 500;
      float lr = 1f;//0.005f;
      for (int i = 0; i < 5; i++) {
         lr = lr / (1.0f + 0.001f * i);
         val timer = Stopwatch.createStarted();
         shuffle = shuffle(shuffle);
         double totalCost = 0;
         double correct = 0;
         float numBatch = 0;
         for (int start = 0; start < shuffle.length; start += batch_size) {
            int[] batch = Arrays.copyOfRange(shuffle, start, Math.min(start + batch_size, shuffle.length));
            float bSize = batch.length;
            if (bSize == 0) {
               continue;
            }
            numBatch++;
            DoubleMatrix X = allX.getColumns(batch);
            DoubleMatrix Y = allY.getColumns(batch);
            //Forward prop
            val z1 = mlp.w1.mmul(X).addiColumnVector(mlp.b1);
            val a1 = MLP.sigmoid(z1);

            //Dropout 0.5
            val mask = DoubleMatrix.rand(a1.length).gt(0.2f);
            a1.muli(mask);

            val z2 = (mlp.w2.mmul(a1)).addiColumnVector(mlp.b2);
            val a2 = MLP.softmax(z2);

            totalCost += -MatrixFunctions.log(a2).mul(Y)
//                                         .add(MatrixFunctions.log(a2.rsub(1.0f)).mul(Y.rsub(1.0f)))
                                         .sum() / bSize;


            correct += correct(a2, Y);

            //backward prop
            val dz2 = a2.sub(Y);
            val dw2 = dz2.mmul(a1.transpose()).divi(bSize);
            val db2 = dz2.rowSums().divi(bSize);
            val dz1 = mlp.w2.transpose().mmul(dz2).muli(MLP.drelu(a1));
//            dz1.muli(mask);
            val dw1 = dz1.mmul(X.transpose()).divi(bSize);
            val db1 = dz1.rowSums().divi(bSize);


            //Weight update
            mlp.w2.subi(dw2.muli(lr));
            mlp.b2.subi(db2.muli(lr));
            mlp.w1.subi(dw1.muli(lr));
            mlp.b1.subi(db1.muli(lr));
         }

         System.out.println("iteration=" + (i + 1) +
                               " loss=" + (totalCost / numBatch) +
                               " accuracy=" + (correct / X_size) +
                               " time=" + timer);
      }

      return mlp;
   }

}// END OF MLPLearner
