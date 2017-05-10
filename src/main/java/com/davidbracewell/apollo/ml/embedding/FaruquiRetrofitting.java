package com.davidbracewell.apollo.ml.embedding;

import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.linalg.store.VectorStore;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.guava.common.collect.Sets;
import com.davidbracewell.guava.common.primitives.Doubles;
import com.davidbracewell.io.resource.Resource;
import com.davidbracewell.string.StringUtils;
import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.eclipse.collections.api.multimap.MutableMultimap;
import org.eclipse.collections.impl.multimap.set.UnifiedSetMultimap;

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
   private final UnifiedSetMultimap<String, String> lexicon = new UnifiedSetMultimap<>();
   @Getter
   @Setter
   private int iterations = 25;


   private void loadLexicon(Resource resource, MutableMultimap<String, String> lexicon) throws IOException {
      resource.forEach(line -> {
         String[] parts = line.toLowerCase().trim().split("\\s+");
         String word = norm(parts[0]);
         for (int i = 1; i < parts.length; i++) {
            lexicon.put(word, norm(parts[i]));
         }
      });
   }

   private String norm(String string) {
      if (Doubles.tryParse(string) != null) {
         return "---num---";
      } else if (StringUtils.isPunctuation(string)) {
         return "---punc---";
      }
      return string.toLowerCase().replace('_', ' ');
   }

   @Override
   public Embedding process(@NonNull Embedding embedding) {
      EncoderPair encoderPair = embedding.getEncoderPair();
      VectorStore<String> origVectors = embedding.getVectorStore();

      Set<String> sourceVocab = new HashSet<>(origVectors.keySet());
      Set<String> sharedVocab = Sets.intersection(sourceVocab, lexicon.keySet().toSet());
      Map<String, Vector> unitNormedVectors = new HashMap<>();
      Map<String, Vector> retrofittedVectors = new HashMap<>();

      //Unit Normalize the vectors
      sourceVocab.forEach(w -> {
         Vector v = origVectors.get(w).toUnitVector();
         retrofittedVectors.put(w, v);
         unitNormedVectors.put(w, v.copy());
      });

      for (int i = 0; i < iterations; i++) {
         sharedVocab.forEach(retrofitTerm -> {
            Set<String> similarTerms = Sets.intersection(lexicon.get(retrofitTerm), sourceVocab);
            if (similarTerms.size() > 0) {
               //Get the original unit normalized vector for the term we are retrofitting
               Vector newTermVector = unitNormedVectors.get(retrofitTerm)
                                                       .mapMultiply(similarTerms.size());

               //Sum the vectors of the similar terms using the retrofitted vectors
               //from last iteration
               similarTerms.forEach(similarTerm -> {
                  newTermVector.addSelf(retrofittedVectors.get(similarTerm));
               });

               //Normalize and update
               double div = 2.0 * similarTerms.size();//v.magnitude() + 1e-6;
               newTermVector.mapDivideSelf(div);
               retrofittedVectors.put(retrofitTerm, newTermVector);
            }
         });
      }

      VectorStore<String> newVectors = embedding.getVectorStore().createNew();
      retrofittedVectors.forEach(newVectors::add);
      return new Embedding(encoderPair, newVectors);
   }

   public void setLexicon(@NonNull Resource resource) throws IOException {
      lexicon.clear();
      loadLexicon(resource, lexicon);
   }

}//END OF FaruquiRetrofitting
