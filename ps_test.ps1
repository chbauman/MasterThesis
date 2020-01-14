# This is a comment

# Array definition
$array = @("val_1","val_2","val_3")

# For loop over array
for ($i=0; $i -lt $array.length; $i++){
    Write-Host $array[$i]
    # Alternatively: echo $array[$i]
}

# Or with for each
foreach ($i in $array){
   Write-Host $i
}
