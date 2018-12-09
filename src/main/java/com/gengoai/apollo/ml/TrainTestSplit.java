package com.gengoai.apollo.ml;

/**
 * @author David B. Bracewell
 */
public class TrainTestSplit {
   public final Dataset train;
   public final Dataset test;

   public TrainTestSplit(Dataset train, Dataset test) {
      this.train = train;
      this.test = test;
   }
}//END OF TrainTest
