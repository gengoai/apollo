package com.gengoai.apollo.ml;

/**
 * Representation of a split (e.g. fold, 80/20, etc.) of a {@link Dataset} into a train and test {@link Dataset}.
 *
 * @author David B. Bracewell
 */
public class Split {
   /**
    * The training dataset
    */
   public final Dataset train;
   /**
    * The testing dataset.
    */
   public final Dataset test;

   /**
    * Instantiates a new Split.
    *
    * @param train the training dataset
    * @param test  the testing dataset.
    */
   public Split(Dataset train, Dataset test) {
      this.train = train;
      this.test = test;
   }


   @Override
   public String toString() {
      return "Split{train=" + train.size() + ", test=" + test.size() + "}";
   }

}//END OF TrainTest
