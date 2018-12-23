package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Example;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.function.SerializableSupplier;

/**
 * @author David B. Bracewell
 */
public class PretrainedVectorizer extends StringVectorizer {
   private static final long serialVersionUID = 1L;
   private final SerializableSupplier<StringVectorizer> supplier;

   public PretrainedVectorizer(SerializableSupplier<StringVectorizer> supplier) {
      super(false);
      this.supplier = supplier;
   }

   @Override
   public String decode(double value) {
      return supplier.get().decode(value);
   }

   @Override
   public double encode(String value) {
      return supplier.get().encode(value);
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public int size() {
      return supplier.get().size();
   }

   @Override
   protected NDArray transformInstance(Example example) {
      return supplier.get().transformInstance(example);
   }

   @Override
   protected NDArray transformSequence(Example example) {
      return supplier.get().transformSequence(example);
   }

}//END OF PretrainedVectorizer
