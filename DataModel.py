from pydantic import BaseModel

class DataModel(BaseModel):

    sdg: float 
    Textos_espanol: str
    #class Config:
      #orm_mode = True

#Esta función retorna los nombres de las columnas correspondientes con el modelo esxportado en joblib.
    def columns(self):
        return ["sgd","Textos_espanol"] 
    