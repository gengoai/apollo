package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.Lazy;
import com.gengoai.apollo.linear.VectorCompositions;
import com.gengoai.apollo.linear.store.VSParameter;
import com.gengoai.apollo.linear.store.VectorStore;

import static com.gengoai.NamedParameters.params;

/**
 * @author David B. Bracewell
 */
public class Embeddings {

   public static final Lazy<StringVectorizer> glove50D = new Lazy<>(Glove50Embedding::new);

   private static class Glove50Embedding extends Embedding {
      private static final long serialVersionUID = 1L;

      public Glove50Embedding() {
         super(VectorStore.builder(params(VSParameter.LOCATION, "/data/Downloads/glove.6B.50d.txt")).build(),
               "unk",
               VectorCompositions.Average);
      }
   }

}//END OF Embeddings
