package com.gengoai.apollo.ml.classification.nn;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.linear.NDArrayInitializer;
import com.gengoai.apollo.ml.optimization.activation.Activation;
import com.gengoai.collection.Lists;

import java.util.List;

/**
 * @author David B. Bracewell
 */
public class DenoisingAutoencoder {

   private NDArray weights;
   private NDArray hbias;
   private NDArray vbias;


   public DenoisingAutoencoder(int nHidden, int nVisible) {
      weights = NDArrayFactory.DEFAULT().create(NDArrayInitializer.glorotAndBengioSigmoid, nHidden, nVisible);
      hbias = NDArrayFactory.DEFAULT().zeros(nHidden);
      vbias = NDArrayFactory.DEFAULT().zeros(nVisible);
   }


   public void train(List<NDArray> X, int batchSize, double learningRate, double corruptionLevel) {
      for (int i = 0; i < 100; i++) {
         for (NDArray x : X) {
            NDArray corruptedInput = x.select(d -> Math.random() >= corruptionLevel);
            NDArray z = weights.mmul(corruptedInput)
                               .addi(hbias)
                               .mapi(Activation.SIGMOID::apply);
            NDArray y = z.T().mmul(weights)
                         .addi(vbias)
                         .mapi(Activation.SIGMOID::apply);

            NDArray vgrad = x.sub(y);
            NDArray hgrad = weights.mmul(x.sub(y)).muli(z.mul(z.rsub(1)));
            NDArray wgrad = hgrad.mmul(corruptedInput.T()).addi(z.mmul(vgrad.T()));

            hbias.addi(hgrad.muli((float) learningRate));
            vbias.addi(vgrad.muli((float) learningRate));
            weights.addi(wgrad.muli((float) learningRate));
         }
      }
      for (NDArray x : X) {
         NDArray z = weights.mmul(x)
                            .addi(hbias)
                            .mapi(Activation.SIGMOID::apply);
         NDArray y = z.T().mmul(weights)
                      .addi(vbias)
                      .mapi(Activation.SIGMOID::apply);
         System.out.println(x);
         System.out.println(y);
         System.out.println();
      }

//         NDArray batch = itr.next();
//
//
//         NDArray mm = weights.mmul(corruptedInput);
//         System.out.println(weights.numRows());
//         System.out.println(weights.numCols());
//
//         NDArray z = mm.addi(hbias, Axis.COlUMN).mapi(Activation.SIGMOID::apply);
//
//         System.out.println(z.numRows());
//         System.out.println(z.numCols());
//
//         NDArray y = z.T().mmul(weights).addi(vbias).mapi(Activation.SIGMOID::apply);
//
//         NDArray v_ = batch.sub(y);
//         NDArray vgrad = v_.sum(Axis.ROW);
//         NDArray h_ = batch.sub(weights.mul(batch.sub(y)))
//                           .muli(z.mul(z.rsubi(1)));
//         NDArray hgrad = h_.sum(Axis.ROW);
//         NDArray wgrad = h_.mul(corruptedInput).addi(v_.mul(z));
//
//         weights.addi(wgrad.mul(learningRate / batchSize));
//         hbias.addi(hgrad.mul(learningRate / batchSize));
//         vbias.addi(vgrad.mul(learningRate / batchSize));
//      }
   }

   public static void main(String[] args) throws Exception {
      List<NDArray> data = Lists.list(
         NDArrayFactory.columnVector(new double[]{1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}),
         NDArrayFactory.columnVector(new double[]{0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0}),
         NDArrayFactory.columnVector(new double[]{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1})
                                     );
      DenoisingAutoencoder autoencoder = new DenoisingAutoencoder(5, 12);
      autoencoder.train(data, 3, 0.1, 0.3);
   }

}//END OF DenoisingAutoencoder
