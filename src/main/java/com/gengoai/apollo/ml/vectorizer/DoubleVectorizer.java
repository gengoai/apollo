package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.Validation;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.json.JsonEntry;

import java.lang.reflect.Type;

/**
 * @author David B. Bracewell
 */
public class DoubleVectorizer implements Vectorizer<Double> {
   private static final long serialVersionUID = 1L;

   public static DoubleVectorizer fromJson(JsonEntry entry, Type... parameters) {
      DoubleVectorizer vectorizer = new DoubleVectorizer();
      Validation.checkState(entry.getStringProperty("class").equalsIgnoreCase(DoubleVectorizer.class.getName()));
      return vectorizer;
   }

   @Override
   public Double decode(double value) {
      return value;
   }

   @Override
   public double encode(Double value) {
      return value;
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public NDArray transform(Example example) {
      NDArray ndArray = NDArrayFactory.DEFAULT().zeros(1, example.size());
      for (int i = 0; i < example.size(); i++) {
         Example c = example.getExample(i);
         if (c.getLabel() instanceof CharSequence) {
            ndArray.set(0, i, Double.parseDouble(c.getLabelAsString()));
         } else {
            ndArray.set(0, i, c.hasLabel() ? c.getLabelAsDouble() : Double.NaN);
         }
      }
      return ndArray;
   }

   @Override
   public int size() {
      return 1;
   }

   public JsonEntry toJson() {
      return JsonEntry.object()
                      .addProperty("class", DoubleVectorizer.class);
   }

   @Override
   public String toString() {
      return "DoubleVectorizer";
   }

}//END OF DoubleVectorizer
