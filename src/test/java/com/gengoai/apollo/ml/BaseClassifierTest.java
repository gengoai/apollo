package com.gengoai.apollo.ml;

import com.gengoai.apollo.ml.classification.Classifier;
import com.gengoai.apollo.ml.classification.ClassifierEvaluation;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.data.DatasetType;
import com.gengoai.apollo.ml.data.format.CSVDataFormat;
import com.gengoai.apollo.ml.data.format.DataFormat;
import com.gengoai.apollo.ml.params.ParamValuePair;
import com.gengoai.io.CSV;
import com.gengoai.io.Resources;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class BaseClassifierTest {

   private final Classifier classifier;
   private final ParamValuePair[] fitParameters;


   public BaseClassifierTest(Classifier classifier,
                             ParamValuePair... fitParameters
                            ) {
      this.classifier = classifier;
      this.fitParameters = fitParameters;
   }

   @Test
   public void fitAndEvaluate() {
      assertTrue(passes(ClassifierEvaluation.crossValidation(irisDataset(),
                                                             classifier,
                                                             10, false, fitParameters)));
   }


   public abstract boolean passes(ClassifierEvaluation mce);

   public Dataset irisDataset() {
      DataFormat csv = new CSVDataFormat(CSV.builder().hasHeader(true), "class");
      try {
         return DatasetType.InMemory.read(Resources.fromClasspath("com/gengoai/apollo/ml/iris.csv"), csv);
      } catch (IOException e) {
         throw new RuntimeException(e);
      }
   }

}//END OF BaseClassifierTest
