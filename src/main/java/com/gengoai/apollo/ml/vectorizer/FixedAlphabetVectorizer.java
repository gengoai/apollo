package com.gengoai.apollo.ml.vectorizer;

import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.collection.Index;
import com.gengoai.collection.Indexes;
import com.gengoai.collection.Sets;

import java.util.Collection;
import java.util.Set;

/**
 * @author David B. Bracewell
 */
public class FixedAlphabetVectorizer extends StringVectorizer {
   private static final long serialVersionUID = 1L;
   final Index<String> alphabet;

   public FixedAlphabetVectorizer(boolean isLabelVectorizer, Collection<String> alphabet) {
      super(isLabelVectorizer);
      this.alphabet = Indexes.indexOf(alphabet);
   }

   @Override
   public Set<String> alphabet() {
      return Sets.asHashSet(alphabet);
   }

   @Override
   public String decode(double value) {
      return alphabet.get((int) value);
   }

   @Override
   public double encode(String value) {
      return alphabet.getId(value);
   }

   @Override
   public void fit(Dataset dataset) {

   }

   @Override
   public int size() {
      return alphabet.size();
   }
}//END OF FixedAlphabetVectorizer
