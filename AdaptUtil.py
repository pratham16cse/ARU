import numpy as np

class AdaptUtil():
    
  def getInitialCombinedStateTupleNeeded(self):
    listT = ()
    for i in range(self._num_states):
      listT = listT + ([],)
    return listT

  def getNumpyArrayFromStates(self,combined_state_tuple_needed):
    listT = ()
    for i in range(self._num_states):
      s_n = combined_state_tuple_needed[i]
      s_n = np.array(s_n)
      listT = listT + (s_n,)
    return listT

  def getAdaptStateFeedDictionary(self,dictionary,combined_state_tuple_needed,adapt_tuple_placeholders):
    for i in range(self._num_states):
      dictionary[adapt_tuple_placeholders[i]] = combined_state_tuple_needed[i]

    return dictionary