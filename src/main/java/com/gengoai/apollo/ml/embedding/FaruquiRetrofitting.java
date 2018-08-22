package com.gengoai.apollo.ml.embedding;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.store.VectorStore;
import com.gengoai.apollo.linear.store.VectorStoreBuilder;
import com.gengoai.collection.Sets;
import com.gengoai.collection.multimap.HashSetMultimap;
import com.gengoai.collection.multimap.Multimap;
import com.gengoai.io.resource.Resource;
import com.gengoai.math.Math2;
import com.gengoai.string.StringUtils;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;

import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * <p>Implementation of <b>Retrofitting Word Vectors to Semantic Lexicons</b> by Faruqui et al.</p>
 *
 * @author David B. Bracewell
 */
public class FaruquiRetrofitting implements Retrofitting {
   private final Multimap<String, String> lexicon = new HashSetMultimap<>();
   @Getter
   @Setter
   private int iterations = 25;


   private void loadLexicon(Resource resource, Multimap<String, String> lexicon) throws IOException {
      resource.forEach(line -> {
         String[] parts = line.toLowerCase().trim().split("\\s+");
         String word = norm(parts[0]);
         for (int i = 1; i < parts.length; i++) {
            lexicon.put(word, norm(parts[i]));
         }
      });
   }

   private String norm(String string) {
      if (Math2.tryParseDouble(string) != null) {
         return "---num---";
      } else if (StringUtils.isPunctuation(string)) {
         return "---punc---";
      }
      return string.toLowerCase().replace('_', ' ');
   }

   @Override
   public Embedding process(@NonNull VectorStore<String> origVectors) {
      Set<String> sourceVocab = new HashSet<>(origVectors.keys());
      Set<String> sharedVocab = Sets.intersection(sourceVocab, lexicon.keySet());
      Map<String, NDArray> unitNormedVectors = new HashMap<>();
      Map<String, NDArray> retrofittedVectors = new HashMap<>();

      //Unit Normalize the vectors
      sourceVocab.forEach(w -> {
         NDArray v = origVectors.get(w).toUnitVector();
         retrofittedVectors.put(w, v);
         unitNormedVectors.put(w, v.copy());
      });

      for (int i = 0; i < iterations; i++) {
         sharedVocab.forEach(retrofitTerm -> {
            Set<String> similarTerms = Sets.intersection(lexicon.get(retrofitTerm), sourceVocab);
            if (similarTerms.size() > 0) {
               //Get the original unit normalized vector for the term we are retrofitting
               NDArray newTermVector = unitNormedVectors.get(retrofitTerm)
                                                        .mul(similarTerms.size());

               //Sum the vectors of the similar terms using the retrofitted vectors
               //from last iteration
               similarTerms.forEach(similarTerm -> {
                  newTermVector.addi(retrofittedVectors.get(similarTerm));
               });

               //Normalize and update
               double div = 2.0 * similarTerms.size();//v.magnitude() + 1e-6;
               newTermVector.divi((float) div);
               retrofittedVectors.put(retrofitTerm, newTermVector);
            }
         });
      }

      VectorStoreBuilder<String> newVectors = origVectors.toBuilder();
      retrofittedVectors.forEach(newVectors::add);
      try {
         return new Embedding(newVectors.build());
      } catch (IOException e) {
         throw new RuntimeException(e);
      }
   }

   public void setLexicon(@NonNull Resource resource) throws IOException {
      lexicon.clear();
      loadLexicon(resource, lexicon);
   }

}//END OF FaruquiRetrofitting
