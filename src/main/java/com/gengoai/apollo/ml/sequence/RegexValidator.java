package com.gengoai.apollo.ml.sequence;

import com.gengoai.apollo.ml.Example;

import java.io.Serializable;
import java.util.regex.Pattern;

/**
 * @author David B. Bracewell
 */
public class RegexValidator implements Validator, Serializable {
   private static final long serialVersionUID = 1L;
   private final Pattern pattern;
   private final boolean inverted;

   public RegexValidator(String pattern) {
      this(false, pattern);
   }


   public RegexValidator(boolean inverted, String... patterns) {
      StringBuilder regex = new StringBuilder();
      for (int i = 0; i < patterns.length; i++) {
         if (i > 0) {
            regex.append('\0');
         }
         regex.append(patterns[i].replace("*", ".*"));
      }
      this.pattern = Pattern.compile(regex.toString());
      this.inverted = inverted;
   }


   @Override
   public boolean isValid(String currentLabel, String previousLabel, Example instance) {
      previousLabel = (previousLabel == null) ? "__BOS__" : previousLabel;
      String lbl = previousLabel + "\0" + currentLabel;
      boolean found = pattern.matcher(lbl).matches();
      if (inverted) {
         return !found;
      }
      return found;
   }
}//END OF RegexValidator
