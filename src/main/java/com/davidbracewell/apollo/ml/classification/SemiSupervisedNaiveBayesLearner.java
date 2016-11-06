package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.MStreamDataset;
import com.davidbracewell.stream.MStream;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.util.concurrent.atomic.AtomicReference;

/**
 * The type Semi supervised naive bayes learner.
 *
 * @author David B. Bracewell
 */
public class SemiSupervisedNaiveBayesLearner extends ClassifierLearner {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter(onParam = @_({@NonNull}))
   private volatile NaiveBayes.ModelType modelType;
   @Getter
   @Setter
   private volatile int maxIterations = 200;

   /**
    * Instantiates a new Semi supervised naive bayes learner.
    */
   public SemiSupervisedNaiveBayesLearner() {
      this.modelType = NaiveBayes.ModelType.Bernoulli;
   }

   /**
    * Instantiates a new Semi supervised naive bayes learner.
    *
    * @param modelType the model type
    */
   public SemiSupervisedNaiveBayesLearner(@NonNull NaiveBayes.ModelType modelType) {
      this.modelType = modelType;
   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      NaiveBayesLearner subLearner = new NaiveBayesLearner(modelType);
      AtomicReference<Classifier> model = new AtomicReference<>(subLearner.trainImpl(dataset));
      for (int iteration = 0; iteration < maxIterations; iteration++) {
         MStream<Instance> mStream = dataset.stream().map(i -> {
            if (!i.hasLabel()) {
               i = i.copy();
               Classification classification = model.get().classify(i);
               i.setLabel(classification.getResult());
               i.setWeight(classification.getConfidence());
            }
            return i;
         });
         model.set(
            subLearner.trainImpl(
               new MStreamDataset<>(dataset.getFeatureEncoder(), dataset.getLabelEncoder(), dataset.getPreprocessors(),
                                    mStream))
                  );
      }
      return model.get();
   }

   @Override
   public void reset() {

   }

}// END OF SemiSupervisedNaiveBayesLearner
